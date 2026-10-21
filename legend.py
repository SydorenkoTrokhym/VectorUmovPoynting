"""HTML for the legend strip above the view and the legend in the data panel."""
from __future__ import annotations

from palette import NEUTRAL, SUM_COLORS, WAVE_COLORS, to_hex

_AXES_NOTE = "<font color='gray'>─ Осі: сірі, підписані X, Y, Z</font>"


def legend_html(tab: int) -> tuple[str, str]:
    """Return (strip, panel) for tab 0 (wave 1), 1 (wave 2) or 2 (superposition)."""
    (e1h, b1h, s1h, _), (e2h, b2h, s2h, _) = (
        tuple(to_hex(color) for color in wave) for wave in WAVE_COLORS)
    axis = to_hex(NEUTRAL)
    axes = (f"<font color='{axis}'>X</font>&nbsp;"
            f"<font color='{axis}'>Y</font>&nbsp;"
            f"<font color='{axis}'>Z</font>")
    if tab == 0:
        strip = (f"<b>Хвиля&nbsp;1:</b>&nbsp;&nbsp;"
                 f"<font color='{e1h}'>■&nbsp;E</font>&nbsp;&nbsp;"
                 f"<font color='{b1h}'>■&nbsp;B</font>&nbsp;&nbsp;"
                 f"<font color='{s1h}'>■&nbsp;S</font>"
                 f"&nbsp;&nbsp;|&nbsp;&nbsp;Осі:&nbsp;{axes}")
        panel = (f"<b>Хвиля 1:</b><br>"
                 f"<font color='{e1h}'>■ E — електричне поле</font><br>"
                 f"<font color='{b1h}'>■ B — магнітне поле</font><br>"
                 f"<font color='{s1h}'>■ S — вектор Пойнтінга</font><br>" + _AXES_NOTE)
    elif tab == 1:
        strip = (f"<b>Хвиля&nbsp;2:</b>&nbsp;&nbsp;"
                 f"<font color='{e2h}'>■&nbsp;E</font>&nbsp;&nbsp;"
                 f"<font color='{b2h}'>■&nbsp;B</font>&nbsp;&nbsp;"
                 f"<font color='{s2h}'>■&nbsp;S</font>"
                 f"&nbsp;&nbsp;|&nbsp;&nbsp;Осі:&nbsp;{axes}")
        panel = (f"<b>Хвиля 2:</b><br>"
                 f"<font color='{e2h}'>■ E — електричне поле</font><br>"
                 f"<font color='{b2h}'>■ B — магнітне поле</font><br>"
                 f"<font color='{s2h}'>■ S — вектор Пойнтінга</font><br>" + _AXES_NOTE)
    else:
        seh, sbh, ssh = (to_hex(color) for color in SUM_COLORS)
        strip = (f"<b>Хвиля&nbsp;1:</b>&nbsp;"
                 f"<font color='{e1h}'>■E₁</font>&nbsp;"
                 f"<font color='{b1h}'>■B₁</font>&nbsp;"
                 f"<font color='{s1h}'>■S₁</font>"
                 f"&nbsp;&nbsp;<b>Хвиля&nbsp;2:</b>&nbsp;"
                 f"<font color='{e2h}'>■E₂</font>&nbsp;"
                 f"<font color='{b2h}'>■B₂</font>&nbsp;"
                 f"<font color='{s2h}'>■S₂</font>"
                 f"&nbsp;&nbsp;|&nbsp;&nbsp;Осі:&nbsp;{axes}"
                 f"&nbsp;&nbsp;<b>Σ:</b>&nbsp;"
                 f"<font color='{seh}'>▬E</font>&nbsp;"
                 f"<font color='{sbh}'>▬B</font>&nbsp;"
                 f"<font color='{ssh}'>▬S</font>")
        panel = (f"<b>Суперпозиція (Хвиля 1 + Хвиля 2):</b><br>"
                 f"<font color='{e1h}'>■</font>/<font color='{e2h}'>■ E — електричне поле</font><br>"
                 f"<font color='{b1h}'>■</font>/<font color='{b2h}'>■ B — магнітне поле</font><br>"
                 f"<font color='{s1h}'>■</font>/<font color='{s2h}'>■ S — вектор Пойнтінга</font><br>"
                 f"<font color='{seh}'>▬</font><font color='{sbh}'>▬</font><font color='{ssh}'>▬</font>"
                 " Σ — сумарне поле в зоні перекриття (окремі хвилі приглушені)<br>"
                 "<font color='gray'>● посередині — точка зустрічі фронтів</font><br>"
                 "<font color='gray'>→ там же: миттєві E₁, B₁, E₂, B₂ та EΣ, BΣ "
                 "у масштабі поля; бліді стрілки — лише напрямки E, B</font><br>"
                 + _AXES_NOTE)
    return strip, panel
