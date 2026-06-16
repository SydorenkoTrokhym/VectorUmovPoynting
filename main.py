import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QComboBox, QDoubleSpinBox,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph.opengl as gl

from physics import calculate_fields, calculate_poynting, calculate_h_field, C, MU0

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Візуалізація вектора Пойнтінга")
        self.resize(1200, 800)

        # Основний віджет та лейаут
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Панель керування (зліва)
        control_panel = QWidget()
        control_panel.setFixedWidth(300)
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel)

        # 3D простір (справа)
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 20
        main_layout.addWidget(self.view)

        # Додавання сітки
        grid = gl.GLGridItem()
        grid.scale(2, 2, 1)
        self.view.addItem(grid)

        # Додавання осей
        self.axes = gl.GLAxisItem()
        self.axes.setSize(12, 12, 12)
        self.view.addItem(self.axes)

        # Елементи керування
        self.init_controls(control_layout)

        # Параметри анімації
        self.t = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)

        # Створення ліній для полів (вектори)
        self.e_line = gl.GLLinePlotItem(width=1, color=(1, 0, 0, 0.6), mode='lines')
        self.b_line = gl.GLLinePlotItem(width=1, color=(0, 0.8, 0, 0.6), mode='lines')
        self.s_line = gl.GLLinePlotItem(width=3, color=(0, 0.4, 1, 1), mode='lines')
        
        # Створення неперервних ліній (оболонки)
        self.e_curve = gl.GLLinePlotItem(width=3, color=(1, 0, 0, 1), mode='line_strip')
        self.b_curve = gl.GLLinePlotItem(width=3, color=(0, 0.8, 0, 1), mode='line_strip')

        self.view.addItem(self.e_line)
        self.view.addItem(self.b_line)
        self.view.addItem(self.s_line)
        self.view.addItem(self.e_curve)
        self.view.addItem(self.b_curve)

        # Створення поверхонь (для Surface representation)
        self.e_surf = gl.GLMeshItem(smooth=True, color=(1, 0, 0, 0.5), shader='shaded', glOptions='additive')
        self.b_surf = gl.GLMeshItem(smooth=True, color=(0, 0.8, 0, 0.5), shader='shaded', glOptions='additive')
        self.s_surf = gl.GLMeshItem(smooth=True, color=(0, 0.4, 1, 0.5), shader='shaded', glOptions='additive')
        self.e_surf.setVisible(False)
        self.b_surf.setVisible(False)
        self.s_surf.setVisible(False)
        self.view.addItem(self.e_surf)
        self.view.addItem(self.b_surf)
        self.view.addItem(self.s_surf)

        # Координати Z
        self.z_steps = 100
        self.z = np.linspace(-10, 10, self.z_steps)
        self.surface_faces = self._build_surface_faces(self.z_steps)

    @staticmethod
    def _build_surface_faces(z_steps):
        faces = []
        for i in range(z_steps - 1):
            idx = i * 2
            # Два трикутники для кожного сегмента
            faces.append([idx, idx + 1, idx + 2])
            faces.append([idx + 1, idx + 3, idx + 2])
        return np.array(faces)

    def init_controls(self, layout):
        # Поляризація
        layout.addWidget(QLabel("Поляризація:"))
        self.pol_combo = QComboBox()
        self.pol_combo.addItems(["linear", "circular_right", "circular_left", "elliptical"])
        layout.addWidget(self.pol_combo)

        # Частота
        layout.addWidget(QLabel("Частота (Гц):"))
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(1e8, 1e10)
        self.freq_spin.setValue(1e9)
        self.freq_spin.setSingleStep(1e8)
        layout.addWidget(self.freq_spin)

        # Амплітуда
        layout.addWidget(QLabel("Амплітуда:"))
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(0.1, 10.0)
        self.amp_spin.setValue(5.0)
        layout.addWidget(self.amp_spin)

        # Кут
        layout.addWidget(QLabel("Кут нахилу (°):"))
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(0, 360)
        self.angle_spin.setValue(0)
        layout.addWidget(self.angle_spin)

        # Тип представлення
        layout.addWidget(QLabel("Представлення:"))
        self.repr_combo = QComboBox()
        self.repr_combo.addItems(["Linear", "Surface"])
        layout.addWidget(self.repr_combo)
        
        # Таблиця даних
        layout.addWidget(QLabel("Легенда:"))
        legend = QLabel("<font color='red'>■ E (Електричне)</font><br>"
                        "<font color='#00CC00'>■ B (Магнітне)</font><br>"
                        "<font color='#0066FF'>■ S (Пойнтінг)</font>")
        layout.addWidget(legend)
        
        layout.addWidget(QLabel("Параметри полів (z=0):"))
        self.data_table = QTableWidget(3, 2)
        self.data_table.setHorizontalHeaderLabels(["Параметр", "Значення"])
        self.data_table.verticalHeader().setVisible(False)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.data_table.setFixedHeight(150)
        
        params = ["|E| (В/м)", "|H| (А/м)", "|S| (Вт/м²)"]
        self.data_value_items = []
        for i, name in enumerate(params):
            self.data_table.setItem(i, 0, QTableWidgetItem(name))
            value_item = QTableWidgetItem("0.00")
            self.data_table.setItem(i, 1, value_item)
            self.data_value_items.append(value_item)
        
        layout.addWidget(self.data_table)
        
        layout.addStretch()

    def update_plot(self):
        self.t += 0.05
        
        freq = self.freq_spin.value()
        amp = self.amp_spin.value()
        pol = self.pol_combo.currentText()
        angle = self.angle_spin.value()
        repr_mode = self.repr_combo.currentText()
        
        Ex, Ey, Bx, By = calculate_fields(self.z, self.t, frequency=freq, amplitude=amp, polarization=pol, angle=angle)
        Sx, Sy, Sz = calculate_poynting(Ex, Ey, Bx, By)
        Hx, Hy = calculate_h_field(Bx, By)

        # Оновлення таблиці (беремо значення при z=0, що приблизно в центрі масиву)
        idx_0 = self.z_steps // 2
        E_norm = np.sqrt(Ex[idx_0]**2 + Ey[idx_0]**2)
        H_norm = np.sqrt(Hx[idx_0]**2 + Hy[idx_0]**2)
        S_norm = np.abs(Sz[idx_0])
        
        self.data_value_items[0].setText(f"{E_norm:.2f}")
        self.data_value_items[1].setText(f"{H_norm:.4f}")
        self.data_value_items[2].setText(f"{S_norm:.2f}")

        # Scale B by c so its display magnitude matches E (|E| = c·|B| in vacuum)
        visible_Bx = Bx * C
        visible_By = By * C

        if repr_mode == "Linear":
            self.e_line.setVisible(True)
            self.b_line.setVisible(True)
            self.e_curve.setVisible(True)
            self.b_curve.setVisible(True)
            self.s_line.setVisible(True)
            self.e_surf.setVisible(False)
            self.b_surf.setVisible(False)
            self.s_surf.setVisible(False)

            # Малюємо вектори як окремі лінії від осі Z
            e_pts = np.zeros((self.z_steps * 2, 3))
            e_pts[0::2, 2] = self.z
            e_pts[1::2, 0] = Ex
            e_pts[1::2, 1] = Ey
            e_pts[1::2, 2] = self.z
            self.e_line.setData(pos=e_pts, mode='lines')

            b_pts = np.zeros((self.z_steps * 2, 3))
            b_pts[0::2, 2] = self.z
            b_pts[1::2, 0] = visible_Bx
            b_pts[1::2, 1] = visible_By
            b_pts[1::2, 2] = self.z
            self.b_line.setData(pos=b_pts, mode='lines')

            # Малюємо неперервну криву по кінцях векторів
            e_curve_pts = np.column_stack([Ex, Ey, self.z])
            self.e_curve.setData(pos=e_curve_pts)

            b_curve_pts = np.column_stack([visible_Bx, visible_By, self.z])
            self.b_curve.setData(pos=b_curve_pts)
        else:
            self.e_line.setVisible(False)
            self.b_line.setVisible(False)
            self.e_curve.setVisible(False)
            self.b_curve.setVisible(False)
            self.s_line.setVisible(False)
            self.e_surf.setVisible(True)
            self.b_surf.setVisible(True)
            self.s_surf.setVisible(True)

            verts_e = np.zeros((self.z_steps * 2, 3))
            verts_e[0::2, 2] = self.z
            verts_e[1::2, 0] = Ex
            verts_e[1::2, 1] = Ey
            verts_e[1::2, 2] = self.z

            verts_b = np.zeros((self.z_steps * 2, 3))
            verts_b[0::2, 2] = self.z
            verts_b[1::2, 0] = visible_Bx
            verts_b[1::2, 1] = visible_By
            verts_b[1::2, 2] = self.z

            self.e_surf.setMeshData(vertexes=verts_e, faces=self.surface_faces)
            self.b_surf.setMeshData(vertexes=verts_b, faces=self.surface_faces)

        # Normalize S so peak display magnitude is ~10 units; amp²/(MU0·C) = 2·<|S|>
        scale_s = 20.0 / (amp**2 / (MU0 * C)) if amp != 0 else 1
        s_val = Sz * scale_s
        
        if repr_mode == "Linear":
            s_pts = np.zeros((self.z_steps * 2, 3))
            s_pts[0::2, 2] = self.z
            s_pts[1::2, 2] = self.z + s_val
            self.s_line.setData(pos=s_pts)
        else:
            verts_s = np.zeros((self.z_steps * 2, 3))
            verts_s[0::2, 2] = self.z
            verts_s[1::2, 2] = self.z + s_val
            self.s_surf.setMeshData(vertexes=verts_s, faces=self.surface_faces)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
