from __future__ import annotations

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def generate_pdf(itinerary, start_date, days):
    output_path = Path("GlobeTrek_Travel_Plan.pdf")

    c = canvas.Canvas(str(output_path), pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(72, 760, "GlobeTrek Travel Plan")
    c.drawString(72, 740, f"Start date: {start_date}")
    c.drawString(72, 722, f"Duration: {days} day(s)")

    y = 690
    for day in itinerary:
        c.drawString(72, y, f"{day['date']} — {day['destination']}")
        y -= 18
        if y < 72:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = 760

    c.save()
    return str(output_path)
