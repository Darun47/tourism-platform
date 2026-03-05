from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


class PDFItineraryGenerator:

    def generate_itinerary_pdf(self, itinerary):

        path = "itinerary.pdf"

        c = canvas.Canvas(path, pagesize=letter)

        y = 750

        for day in itinerary["days"]:

            c.drawString(100, y, f"City: {day['city']}")
            y -= 20

            c.drawString(100, y, f"Site: {day['site']}")
            y -= 20

            c.drawString(100, y, f"Cost: ${day['cost']}")
            y -= 40

        c.save()

        return path
