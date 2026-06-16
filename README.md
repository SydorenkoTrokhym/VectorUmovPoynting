# VectorUmovPoynting - Docker Setup

Цей проект візуалізує вектор Пойнтінга у 3D за допомогою PyQt6 та PyQtGraph.
Оскільки це GUI-додаток, запуск у Docker потребує налаштування X11 Forwarding.

## Підготовка

### Linux
1. Дозвольте підключення до X-сервера:
   ```bash
   xhost +local:docker
   ```

### Windows (з використанням VcXsrv або Xming)
1. Встановіть [VcXsrv](https://sourceforge.net/projects/vcxsrv/).
2. Запустіть `XLaunch` з налаштуваннями:
   - "Multiple windows"
   - "Display number" = 0
   - **Важливо:** Увімкніть "Disable access control".

## Збірка та запуск

1. **Зберіть образ:**
   ```bash
   docker build -t poynting-viz .
   ```

2. **Запустіть контейнер:**

   **Linux:**
   ```bash
   docker run -it --rm \
     -e DISPLAY=$DISPLAY \
     -v /tmp/.X11-unix:/tmp/.X11-unix \
     poynting-viz
   ```

   **Windows (PowerShell):**
   ```powershell
   # Замініть <YOUR_IP> на IP-адресу вашого хоста (не 127.0.0.1)
   docker run -it --rm `
     -e DISPLAY=<YOUR_IP>:0 `
     poynting-viz
   ```

## Примітка щодо OpenGL
Якщо у вас виникають помилки ініціалізації OpenGL у контейнері, спробуйте додати прапорець `--device /dev/dri:/dev/dri` (тільки для Linux з підтримкою GPU) або скористатися програмним рендерингом, встановивши `LIBGL_ALWAYS_SOFTWARE=1`.
