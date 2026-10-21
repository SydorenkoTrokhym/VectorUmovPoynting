from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox,
                              QHBoxLayout, QHeaderView, QLabel, QLayout,
                              QPushButton, QSpinBox, QTableWidget, QVBoxLayout,
                              QWidget)

from camera import BG_COLORS
from models import DEFAULT_SOURCE_SEPARATION
from persistence import SEPARATION_RANGE


def build_controls(window, layout: QLayout) -> None:
    """Build every control widget and wire it to the window's slot handlers.

    Keeping this entry point separate prevents application startup and tests from
    depending on the concrete control layout.
    """
    # Visibility controls
    vis_ctrl_group = QGroupBox("Відображення панелей")
    vis_ctrl_layout = QVBoxLayout()

    window.chk_wave = QCheckBox("Параметри хвилі")
    window.chk_viz = QCheckBox("Візуалізація")
    window.chk_mod = QCheckBox("Модуляція")
    window.chk_data = QCheckBox("Дані та легенда")

    # Initial states (mimic "Basic parameters" set)
    window.chk_wave.setChecked(True)
    window.chk_viz.setChecked(False)
    window.chk_mod.setChecked(False)
    window.chk_data.setChecked(False)

    for chk in [window.chk_wave, window.chk_viz, window.chk_mod, window.chk_data]:
        chk.stateChanged.connect(window._on_visibility_changed)
        vis_ctrl_layout.addWidget(chk)

    vis_ctrl_group.setLayout(vis_ctrl_layout)
    layout.addWidget(vis_ctrl_group)

    # Wave parameters
    window.wave_group = QGroupBox("Параметри хвилі")
    wave_layout = QVBoxLayout()

    wave_layout.addWidget(QLabel("Поляризація:"))
    window.pol_combo = QComboBox()
    window.pol_combo.addItems(["linear", "circular_right", "circular_left", "elliptical"])
    window.pol_combo.currentTextChanged.connect(window._on_param_changed)
    wave_layout.addWidget(window.pol_combo)

    wave_layout.addWidget(QLabel("Амплітуда E = A (В/м):"))
    window.amp_spin = QDoubleSpinBox()
    window.amp_spin.setRange(0.1, 100.0)
    window.amp_spin.setValue(1.0)
    window.amp_spin.setSingleStep(0.5)
    window.amp_spin.valueChanged.connect(window._on_param_changed)
    wave_layout.addWidget(window.amp_spin)

    wave_layout.addWidget(QLabel("Зсув по фазі (°):"))
    phase_row = QHBoxLayout()
    window.phase_spin = QDoubleSpinBox()
    window.phase_spin.setRange(0, 360)
    window.phase_spin.setValue(0)
    window.phase_spin.valueChanged.connect(window._on_param_changed)
    phase_row.addWidget(window.phase_spin)
    window.phase_preset_combo = QComboBox()
    window.phase_preset_combo.addItems(["Custom", "0", "90", "180", "270"])
    window.phase_preset_combo.currentIndexChanged.connect(window._on_phase_preset_changed)
    phase_row.addWidget(window.phase_preset_combo)
    wave_layout.addLayout(phase_row)

    wave_layout.addWidget(QLabel("Довжина відображуваного відрізка (λ):"))
    window.n_periods_spin = QSpinBox()
    window.n_periods_spin.setRange(1, 10)
    window.n_periods_spin.setValue(2)
    window.n_periods_spin.valueChanged.connect(window._on_param_changed)
    wave_layout.addWidget(window.n_periods_spin)

    wave_layout.addWidget(QLabel("Напрямок:"))
    window.dir_mode_combo = QComboBox()
    window.dir_mode_combo.addItems(["Базова вісь", "Довільний вектор"])
    window.dir_mode_combo.currentTextChanged.connect(window._on_dir_mode_changed)
    wave_layout.addWidget(window.dir_mode_combo)

    window.wave_prop_dir_combo = QComboBox()
    window.wave_prop_dir_combo.addItems(["+Z", "-Z", "+X", "-X", "+Y", "-Y"])
    window.wave_prop_dir_combo.currentTextChanged.connect(window._on_propagation_changed)
    wave_layout.addWidget(window.wave_prop_dir_combo)

    window.dir_vec_widget = QWidget()
    dv_layout = QHBoxLayout(window.dir_vec_widget)
    dv_layout.setContentsMargins(0, 0, 0, 0)
    for lbl_txt, attr, default in [("X:", 'dir_x_spin', 0.), ("Y:", 'dir_y_spin', 0.), ("Z:", 'dir_z_spin', 1.)]:
        dv_layout.addWidget(QLabel(lbl_txt))
        sp = QDoubleSpinBox()
        sp.setRange(-5.0, 5.0); sp.setSingleStep(0.1); sp.setDecimals(1); sp.setValue(default)
        sp.setFixedWidth(52)
        sp.valueChanged.connect(window._on_param_changed)
        setattr(window, attr, sp)
        dv_layout.addWidget(sp)
    window.dir_vec_widget.setVisible(False)
    wave_layout.addWidget(window.dir_vec_widget)

    wave_layout.addWidget(QLabel("Початкові напрямки полів:"))
    field_direction_row = QHBoxLayout()
    field_direction_row.addWidget(QLabel("E:"))
    window.e_dir_combo = QComboBox()
    window.e_dir_combo.addItems(["+X", "-X", "+Y", "-Y"])
    window.e_dir_combo.currentTextChanged.connect(window._on_field_direction_changed)
    field_direction_row.addWidget(window.e_dir_combo)
    field_direction_row.addWidget(QLabel("B:"))
    window.b_dir_combo = QComboBox()
    window.b_dir_combo.addItems(["+Y", "-Y"])
    window.b_dir_combo.currentTextChanged.connect(window._on_field_direction_changed)
    field_direction_row.addWidget(window.b_dir_combo)
    wave_layout.addLayout(field_direction_row)
    window.field_direction_status = QLabel("E ⊥ B;  E × B = +Z")
    window.field_direction_status.setStyleSheet("font-size: 8pt; color: #33aa55;")
    wave_layout.addWidget(window.field_direction_status)

    window.wave_group.setLayout(wave_layout)
    layout.addWidget(window.wave_group)

    # Source position (Tab 2 only — direction is in wave_group above)
    window.source_group = QGroupBox("Параметри взаємодії (Суперпозиція)")
    src_layout = QVBoxLayout()

    window.source_position_widget = QWidget()
    source_position_layout = QVBoxLayout(window.source_position_widget)
    source_position_layout.setContentsMargins(0, 0, 0, 0)
    source_position_layout.addWidget(QLabel("Координати джерела 2 (×λ):"))
    pos_row2 = QHBoxLayout()
    for lbl, attr, default_val in [("X:", 'x0_spin', 0.0), ("Y:", 'y0_spin', 0.0), ("Z:", 'z0_spin', 2.0)]:
        pos_row2.addWidget(QLabel(lbl))
        spin = QDoubleSpinBox()
        spin.setRange(-10.0, 10.0)
        spin.setValue(default_val)
        spin.setSingleStep(0.25)
        spin.setFixedWidth(70)
        spin.valueChanged.connect(window._on_source_changed)
        setattr(window, attr, spin)
        pos_row2.addWidget(spin)
    source_position_layout.addLayout(pos_row2)
    src_layout.addWidget(window.source_position_widget)

    # Superposition only: distance between the sources
    window.separation_widget = QWidget()
    separation_layout = QHBoxLayout(window.separation_widget)
    separation_layout.setContentsMargins(0, 0, 0, 0)
    separation_layout.addWidget(QLabel("Відстань між джерелами:"))
    window.separation_spin = QDoubleSpinBox()
    window.separation_spin.setRange(*SEPARATION_RANGE)
    window.separation_spin.setSingleStep(0.5)
    window.separation_spin.setDecimals(2)
    window.separation_spin.setSuffix(" λ")
    window.separation_spin.setValue(DEFAULT_SOURCE_SEPARATION)
    window.separation_spin.setToolTip(
        "Джерело 1 — у початку координат; джерело 2 — у напрямку, заданому на вкладці "
        "«Хвиля 2». Фронти зустрічаються через половину цієї відстані в періодах T.")
    window.separation_spin.valueChanged.connect(window._on_separation_changed)
    separation_layout.addWidget(window.separation_spin)
    window.separation_widget.setVisible(False)
    src_layout.addWidget(window.separation_widget)

    # Superposition only: jump to characteristic moments after both sources switch on
    window.superposition_time_widget = QWidget()
    time_layout = QVBoxLayout(window.superposition_time_widget)
    time_layout.setContentsMargins(0, 0, 0, 0)
    time_layout.addWidget(QLabel("Момент часу (джерела вмикаються при t = 0):"))
    preset_row = QHBoxLayout()
    for text, tooltip, preset in [
        ("Старт", "Фронти щойно вийшли з джерел — видно орієнтацію джерел і напрямки хвиль", "start"),
        ("Зустріч", "Фронти зустрічаються посередині між джерелами", "meet"),
        ("Пізніше", "Хвилі перекриваються на всьому відрізку — сумарне поле", "later"),
    ]:
        button = QPushButton(text)
        button.setToolTip(tooltip)
        button.clicked.connect(lambda _checked=False, p=preset: window._on_time_preset(p))
        preset_row.addWidget(button)
    time_layout.addLayout(preset_row)
    step_row = QHBoxLayout()
    for text, sign in [("◀ −T/20", -1), ("+T/20 ▶", 1)]:
        button = QPushButton(text)
        button.clicked.connect(lambda _checked=False, s=sign: window._on_time_step(s))
        step_row.addWidget(button)
    time_layout.addLayout(step_row)
    window.time_status = QLabel("")
    window.time_status.setWordWrap(True)
    window.time_status.setStyleSheet("font-size: 8pt;")
    time_layout.addWidget(window.time_status)
    window.superposition_time_widget.setVisible(False)
    src_layout.addWidget(window.superposition_time_widget)

    window.source_group.setLayout(src_layout)
    window.source_group.setVisible(False)
    layout.addWidget(window.source_group)

    # Visualization
    window.viz_group = QGroupBox("Візуалізація")
    viz_layout = QVBoxLayout()

    viz_layout.addWidget(QLabel("Проекція:"))
    window.proj_combo = QComboBox()
    window.proj_combo.addItems(["3D", "XY (Top)", "XZ (Side)", "YZ (Front)"])
    window.proj_combo.currentIndexChanged.connect(window._on_projection_changed)
    viz_layout.addWidget(window.proj_combo)

    viz_layout.addWidget(QLabel("Пікселів на довжину хвилі:"))
    window.pxw_spin = QSpinBox()
    window.pxw_spin.setRange(10, 2000)
    window.pxw_spin.setValue(100)
    window.pxw_spin.setSingleStep(10)
    window.pxw_spin.valueChanged.connect(window._on_pxw_changed)
    viz_layout.addWidget(window.pxw_spin)

    viz_layout.addWidget(QLabel("Відображення компонент:"))
    window.comp_combo = QComboBox()
    window.comp_combo.addItems(["Узагальнений підхід", "Електрична (E)", "Магнітна (B)"])
    window.comp_combo.currentTextChanged.connect(window._on_repr_changed)
    viz_layout.addWidget(window.comp_combo)

    viz_layout.addWidget(QLabel("Представлення:"))
    window.repr_combo = QComboBox()
    window.repr_combo.addItems(["Linear (3D)", "Surface (3D)"])
    window.repr_combo.currentTextChanged.connect(window._on_repr_changed)
    viz_layout.addWidget(window.repr_combo)

    viz_layout.addWidget(QLabel("Фон:"))
    window.bg_combo = QComboBox()
    window.bg_combo.addItems(list(BG_COLORS.keys()))
    window.bg_combo.currentTextChanged.connect(window._apply_background)
    viz_layout.addWidget(window.bg_combo)

    window.viz_group.setLayout(viz_layout)
    layout.addWidget(window.viz_group)

    # Modulation
    window.mod_group = QGroupBox("Модуляція")
    mod_layout = QVBoxLayout()

    mod_layout.addWidget(QLabel("Тип модуляції:"))
    window.mod_combo = QComboBox()
    window.mod_combo.addItems(["none", "AM", "FM", "AFM"])
    window.mod_combo.currentTextChanged.connect(window._on_param_changed)
    mod_layout.addWidget(window.mod_combo)

    mod_layout.addWidget(QLabel("Частота модуляції (Гц):"))
    window.mod_freq_spin = QDoubleSpinBox()
    window.mod_freq_spin.setRange(0.01, 1e9)
    window.mod_freq_spin.setValue(0.1)
    window.mod_freq_spin.setSingleStep(0.1)
    window.mod_freq_spin.valueChanged.connect(window._on_param_changed)
    mod_layout.addWidget(window.mod_freq_spin)

    mod_layout.addWidget(QLabel("Індекс модуляції:"))
    window.mod_index_spin = QDoubleSpinBox()
    window.mod_index_spin.setRange(0.0, 1.0)
    window.mod_index_spin.setValue(0.5)
    window.mod_index_spin.setSingleStep(0.1)
    window.mod_index_spin.valueChanged.connect(window._on_param_changed)
    mod_layout.addWidget(window.mod_index_spin)

    window.mod_group.setLayout(mod_layout)
    layout.addWidget(window.mod_group)

    # Data & legend
    window.data_group = QGroupBox("Дані та легенда")
    data_layout = QVBoxLayout()

    window.legend_label = QLabel("")   # filled by _update_legend
    data_layout.addWidget(window.legend_label)

    data_layout.addWidget(QLabel("Компоненти полів (період λ/10):"))
    window.data_table = QTableWidget(11, 6)
    window.data_table.setHorizontalHeaderLabels(["Coord", "E₁", "E₂", "H₁", "H₂", "Sz"])

    vh = window.data_table.verticalHeader()
    if vh is not None:
        vh.setVisible(False)
        vh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    hh = window.data_table.horizontalHeader()
    if hh is not None:
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)

    window.data_table.setFixedHeight(350)
    window.data_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    window.data_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    data_layout.addWidget(window.data_table)
    window.data_group.setLayout(data_layout)
    layout.addWidget(window.data_group)

    window.pause_button = QPushButton("⏸ Пауза")
    window.pause_button.setCheckable(True)
    window.pause_button.setChecked(True)
    window.pause_button.toggled.connect(window._on_pause_toggled)
    window.pause_button.setText("▶ Відтворити")
    layout.addWidget(window.pause_button)

    window.reset_button = QPushButton("⟲ Повернутись до початкових налаштувань")
    window.reset_button.clicked.connect(window._on_reset)
    layout.addWidget(window.reset_button)

    window.save_button = QPushButton("💾 Зберегти малюнок")
    window.save_button.clicked.connect(window._on_save_image)
    layout.addWidget(window.save_button)

    # Parameters save / load
    params_io_group = QGroupBox("Файл параметрів")
    params_io_layout = QVBoxLayout()
    io_row = QHBoxLayout()
    window.save_params_btn = QPushButton("💾 Зберегти")
    window.load_params_btn = QPushButton("📂 Завантажити")
    window.save_params_btn.clicked.connect(window._on_save_params)
    window.load_params_btn.clicked.connect(window._on_load_params)
    io_row.addWidget(window.save_params_btn)
    io_row.addWidget(window.load_params_btn)
    params_io_layout.addLayout(io_row)
    window.params_status = QLabel("—")
    window.params_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
    window.params_status.setWordWrap(True)
    window.params_status.setStyleSheet("font-size: 8pt; color: gray;")
    params_io_layout.addWidget(window.params_status)
    params_io_group.setLayout(params_io_layout)
    layout.addWidget(params_io_group)

    window.exit_button = QPushButton("Вихід")
    window.exit_button.clicked.connect(window.close)
    layout.addWidget(window.exit_button)
    layout.addStretch()