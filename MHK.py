import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget, QSlider, QLabel
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "2R Robotic Arm Inverse Kinematics"
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)

        self.canvas = MplCanvas()

        # Sliders for end effector X and Y positions
        self.slider_x, self.label_x = self.create_slider_with_label('End Effector X:', -600, 600, 200, self.update_plot)
        self.slider_y, self.label_y = self.create_slider_with_label('End Effector Y:', -600, 600, 200, self.update_plot)
        # Sliders for link lengths L1 and L2
        self.slider_L1, self.label_L1 = self.create_slider_with_label('L1:', 1, 600, 200, self.update_plot)
        self.slider_L2, self.label_L2 = self.create_slider_with_label('L2:', 1, 600, 200, self.update_plot)

        layout = QVBoxLayout()
        layout.addWidget(self.label_x)
        layout.addWidget(self.slider_x)
        layout.addWidget(self.label_y)
        layout.addWidget(self.slider_y)
        layout.addWidget(self.label_L1)
        layout.addWidget(self.slider_L1)
        layout.addWidget(self.label_L2)
        layout.addWidget(self.slider_L2)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.update_plot()

    def create_slider_with_label(self, label, min_val, max_val, init_val, callback):
        slider_label = QLabel(f'{label} {init_val:.2f}')
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(init_val)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)
        slider.valueChanged.connect(lambda value, lbl=label: slider_label.setText(f'{lbl} {value / 100.0:.2f}'))
        slider.valueChanged.connect(lambda _: callback())
        return slider, slider_label

    def update_plot(self):
        x = self.slider_x.value() / 100.0
        y = self.slider_y.value() / 100.0
        L1 = self.slider_L1.value() / 100.0
        L2 = self.slider_L2.value() / 100.0

        theta1, theta2 = self.inverse_kinematics(x, y, L1, L2)

        if theta1 is not None and theta2 is not None:
            self.plot_robot_arm(theta1, theta2, L1, L2)
        else:
            # Handle the case where the inverse kinematics cannot be solved
            self.canvas.ax.clear()
            self.canvas.ax.text(0.5, 0.5, 'Solution not possible\nfor the given x, y', horizontalalignment='center', verticalalignment='center', transform=self.canvas.ax.transAxes)
            self.canvas.ax.text(0.5, 0.05, 'Copyrighted by MHK', horizontalalignment='center', verticalalignment='center', transform=self.canvas.ax.transAxes, fontsize=8)
            self.canvas.draw()

    def inverse_kinematics(self, x, y, L1, L2):
        # Inverse Kinematics calculations to find theta1 and theta2
        # based on the provided x, y, L1, and L2
        D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)

        if abs(D) > 1:
            return None, None  # No solution

        theta2 = np.arctan2(np.sqrt(1 - D**2), D)
        theta1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))

        return np.degrees(theta1), np.degrees(theta2)

    def plot_robot_arm(self, theta1, theta2, L1, L2):
        theta1_rad = np.radians(theta1)
        theta2_rad = np.radians(theta2)

        joint1_x = L1 * np.cos(theta1_rad)
        joint1_y = L1 * np.sin(theta1_rad)

        end_effector_x = joint1_x + L2 * np.cos(theta1_rad + theta2_rad)
        end_effector_y = joint1_y + L2 * np.sin(theta1_rad + theta2_rad)

        self.canvas.ax.clear()
        self.canvas.ax.plot([0, joint1_x, end_effector_x], [0, joint1_y, end_effector_y], 'k-')
        self.canvas.ax.plot([0, joint1_x], [0, joint1_y], 'bo', markersize=10)
        self.canvas.ax.plot(end_effector_x, end_effector_y, 'ro', markersize=10)
        self.canvas.ax.text(end_effector_x, end_effector_y, f'θ1: {theta1:.2f}°\nθ2: {theta2:.2f}°', fontsize=8)
        self.canvas.ax.set_xlim(-L1 - L2 - 1, L1 + L2 + 1)
        self.canvas.ax.set_ylim(-L1 - L2 - 1, L1 + L2 + 1)
        self.canvas.ax.set_title('2R Robotic Arm Configuration')
        self.canvas.ax.grid(True)
        self.canvas.ax.text(0.5, 0.05, 'Copyrighted by MHK', horizontalalignment='center', verticalalignment='center', transform=self.canvas.ax.transAxes, fontsize=8)
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Window()
    main_window.show()
    sys.exit(app.exec_())
