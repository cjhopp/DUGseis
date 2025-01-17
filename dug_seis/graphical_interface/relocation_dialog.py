# DUGSeis
# Copyright (C) 2021 DUGSeis Authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
The relocation window.
"""

import collections

import matplotlib.pyplot as plt
import numpy as np
import obspy
from PySide6 import QtGui, QtCore, QtWidgets

from ..project.project import DUGSeisProject
from .ui_files.relocation_window import Ui_RelocationWindow


class RelocationDialog(QtWidgets.QDialog):
    """
    Relocation helper window.

    Can currently only use P picks, one P pick per channel.
    """

    def __init__(self, parent, event: obspy.core.event.Event, project: DUGSeisProject):
        super().__init__(parent)
        self.ui = Ui_RelocationWindow()
        self.ui.setupUi(self)
        self.parent = parent

        self.event = event
        self.new_origin = None
        self.project = project

        c = self.project.config["graphical_interface"][
            "location_algorithm_default_args"
        ]
        self.ui.velocity_spin_box.setValue(c["velocity"])
        self.ui.damping_spin_box.setValue(c["damping"])
        self.ui.use_anisotropy_check_box.setChecked(c["use_anisotropy"])
        self.ui.azi_spin_box.setValue(c["azi"])
        self.ui.inc_spin_box.setValue(c["inc"])
        self.ui.delta_spin_box.setValue(c["delta"])
        self.ui.epsilon_spin_box.setValue(c["epsilon"])

        # Figure out the initial picks to use. Filter by P pick and also filter
        # by channel id.
        picks_by_channel = collections.defaultdict(list)
        for pick in self.event.picks:
            if pick.phase_hint and pick.phase_hint.lower() != "p":
                continue
            picks_by_channel[pick.waveform_id.id].append(pick)

        selected_picks = []

        for key, picks in picks_by_channel.items():
            if len(picks) == 1:
                selected_picks.append(picks[0])
                continue

            manual_picks = []
            automatic_picks = []

            for p in picks:
                if p.evaluation_mode == "manual":
                    manual_picks.append(p)
                else:
                    automatic_picks.append(p)

            chosen_picks = manual_picks if manual_picks else automatic_picks
            selected_picks.append(chosen_picks[0])

        self.pick_info = []
        for p in selected_picks:
            self.pick_info.append(
                {
                    "pick": p,
                    "active": True,
                    "distance": None,
                    "rms": None,
                    "rms_per_meter": None,
                    "phase_hint": p.phase_hint,
                    "evaluation_mode": p.evaluation_mode,
                }
            )

        self.ui.pick_table.setRowCount(len(self.pick_info))
        self.ui.pick_table.setColumnCount(7)
        self.ui.pick_table.setHorizontalHeaderLabels(
            [
                "Use Pick",
                "Channel",
                "Distance [m]",
                "Time error [ms]",
                "Time error/distance [ms/m]",
                "Phase hint",
                "Evaluation mode",
            ]
        )
        header = self.ui.pick_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(6, QtWidgets.QHeaderView.Stretch)

        self.update_table_in_gui()

    def update_table_in_gui(self):
        for row, info in enumerate(self.pick_info):
            # Checkbox.
            cb = QtWidgets.QTableWidgetItem()
            cb.setCheckState(
                QtCore.Qt.CheckState.Checked
                if info["active"]
                else QtCore.Qt.CheckState.Unchecked
            )
            self.ui.pick_table.setItem(row, 0, cb)

            # id.
            self.ui.pick_table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(info["pick"].waveform_id.id)
            )

            # Distance.
            self.ui.pick_table.setItem(
                row, 2, QtWidgets.QTableWidgetItem(str(info["distance"]))
            )

            # error per meter
            self.ui.pick_table.setItem(
                row, 3, QtWidgets.QTableWidgetItem(str(info["rms"]))
            )

            # error per meter.
            self.ui.pick_table.setItem(
                row, 4, QtWidgets.QTableWidgetItem(str(info["rms_per_meter"]))
            )

            # Phase hint.
            self.ui.pick_table.setItem(
                row, 5, QtWidgets.QTableWidgetItem(str(info["phase_hint"]))
            )

            # Evaluation mode.
            self.ui.pick_table.setItem(
                row, 6, QtWidgets.QTableWidgetItem(str(info["evaluation_mode"]))
            )

    def update_pick_info_from_table(self):
        row_count = self.ui.pick_table.rowCount()
        assert row_count == len(self.pick_info)

        # The only thing to update is the check state.
        for i in range(row_count):
            self.pick_info[i]["active"] = (
                self.ui.pick_table.item(i, 0).checkState()
                == QtCore.Qt.CheckState.Checked
            )

    def setCentralWidget(self, *args, **kwargs):
        pass

    @QtCore.Slot()
    def on_relocate_push_button_released(self):
        from dug_seis.event_processing.location.locate_homogeneous import (
            locate_in_homogeneous_background_medium,
        )

        self.update_pick_info_from_table()

        picks = [p["pick"] for p in self.pick_info if p["active"]]

        if self.ui.use_anisotropy_check_box.isChecked():
            anisotropic_params = {
                "azi": self.ui.azi_spin_box.value(),
                "inc": self.ui.inc_spin_box.value(),
                "delta": self.ui.delta_spin_box.value(),
                "epsilon": self.ui.epsilon_spin_box.value(),
            }
        else:
            anisotropic_params = {}

        event = locate_in_homogeneous_background_medium(
            picks=picks,
            coordinates=self.project.cartesian_coordinates,
            velocity=self.ui.velocity_spin_box.value(),
            damping=self.ui.damping_spin_box.value(),
            local_to_global_coordinates=self.project.local_to_global_coordinates,
            anisotropic_params=anisotropic_params,
        )

        assert len(event.origins) == 1

        origin = event.origins[0]
        event_location = self.project.global_to_local_coordinates(
            latitude=origin.latitude, longitude=origin.longitude, depth=origin.depth
        )

        assert len(origin.arrivals) == len(picks)

        pick_arr = {}
        for p in picks:
            arr = [a for a in origin.arrivals if a.pick_id == p.resource_id]
            assert len(arr) == 1
            pick_arr[p.waveform_id.id] = {"pick": p, "arrival": arr[0]}

        # Now update the pick info.
        row_count = self.ui.pick_table.rowCount()

        # The only thing to update is the check state.

        time_residuals = []
        time_distance_residuals = []

        for i in range(row_count):
            waveform_id = self.ui.pick_table.item(i, 1).text()
            if waveform_id not in pick_arr:
                self.ui.pick_table.item(i, 2).setText("--")
                self.ui.pick_table.item(i, 3).setText("--")
                self.ui.pick_table.item(i, 4).setText("--")

                time_residuals.append(0.0)
                time_distance_residuals.append(0.0)
            else:
                distance = np.linalg.norm(
                    self.project.channels[waveform_id]["coordinates"] - event_location
                )
                self.ui.pick_table.item(i, 2).setText(f"{distance:.2f}")

                # Convert to ms.
                e = pick_arr[waveform_id]["arrival"]["time_residual"] * 1000.0
                self.ui.pick_table.item(i, 3).setText(f"{e:.5f}")

                self.ui.pick_table.item(i, 4).setText(f"{e / distance:.5f}")
                time_residuals.append(abs(e))
                time_distance_residuals.append(abs(e) / distance)

        # Color things.
        cmap = plt.get_cmap("RdYlGn_r")
        for i, (time_residual, time_distance_residual) in enumerate(
            zip(time_residuals, time_distance_residuals)
        ):
            if time_residual == 0.0 and time_distance_residual == 0.0:
                self.ui.pick_table.item(i, 3).setBackground(QtGui.QColor("white"))
                self.ui.pick_table.item(i, 4).setBackground(QtGui.QColor("white"))
                continue

            tr_color = cmap(time_residual / max(time_residuals))
            tr_d_color = cmap(time_distance_residual / max(time_distance_residuals))

            self.ui.pick_table.item(i, 3).setBackground(
                QtGui.QColor(
                    int(tr_color[0] * 255),
                    int(tr_color[1] * 255),
                    int(tr_color[2] * 255),
                    100,
                )
            )
            self.ui.pick_table.item(i, 4).setBackground(
                QtGui.QColor(
                    int(tr_d_color[0] * 255),
                    int(tr_d_color[1] * 255),
                    int(tr_d_color[2] * 255),
                    100,
                )
            )

        rms = origin.time_errors.uncertainty * 1000.0
        self.ui.rms_time_error_label.setText(f"{rms:.5f}")
        self.new_origin = origin

    @QtCore.Slot()
    def on_save_new_origin_push_button_released(self):
        if not self.new_origin:
            print("No new origin")
            return

        # Add the origin to the event and set it to the new preferred origin.
        self.event.origins.append(self.new_origin)
        self.event.preferred_origin_id = self.new_origin.resource_id

        # Save and update.
        self.project.db.update_event(self.event)
        event_number = self.parent.ui.event_number_spin_box.value()
        self.parent.load_event(event_number=event_number)

        self.done(1)
