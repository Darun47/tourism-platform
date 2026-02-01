"""
PDF Itinerary Generation Module
================================

Generates professional PDF itineraries with:
- Cover page with trip details
- Daily schedule with sites and activities
- Cost breakdown
- Packing list and recommendations
- Weather information

Dependencies: reportlab
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
from typing import Dict, Any
import io

class PDFItineraryGenerator:
    """Generate professional PDF itineraries"""
    
    def __init__(self):
        """Initialize PDF generator"""
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495E'),
            spaceAfter=12,
            fontName='Helvetica-Bold'
        ))
        
        # Day heading style
        self.styles.add(ParagraphStyle(
            name='DayHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#E74C3C'),
            spaceAfter=10,
            fontName='Helvetica-Bold'
        ))
        
        # Body text
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor('#2C3E50'),
            alignment=TA_JUSTIFY
        ))
    
    def generate_itinerary_pdf(
        self,
        itinerary_data: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Generate complete PDF itinerary
        
        Args:
            itinerary_data: Itinerary dictionary from backend engine
            output_path: Path to save PDF file
            
        Returns:
            Path to generated PDF
        """
        print(f"📄 Generating PDF itinerary...")
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Add content
        elements.extend(self._create_cover_page(itinerary_data))
        elements.append(PageBreak())
        elements.extend(self._create_itinerary_details(itinerary_data))
        elements.append(PageBreak())
        elements.extend(self._create_cost_breakdown(itinerary_data))
        elements.extend(self._create_recommendations(itinerary_data))
        
        # Build PDF
        doc.build(elements)
        
        print(f"✓ PDF generated: {output_path}")
        return output_path
    
    def _create_cover_page(self, data: Dict[str, Any]) -> list:
        """Create cover page"""
        elements = []
        
        # Title
        title = Paragraph(
            "Your Personalized Travel Itinerary",
            self.styles['CustomTitle']
        )
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))
        
        # Trip details
        itinerary = data['itinerary']
        tourist = data['tourist_profile']
        
        details_data = [
            ['Destination(s):', ', '.join(itinerary['cities_visited'])],
            ['Travel Dates:', f"{itinerary['start_date']} to {itinerary['end_date']}"],
            ['Duration:', f"{itinerary['total_days']} days"],
            ['Total Budget:', f"${itinerary['total_cost_usd']:,.2f}"],
            ['Daily Average:', f"${itinerary['avg_daily_cost_usd']:,.2f}"],
            ['Budget Level:', tourist['budget']],
        ]
        
        details_table = Table(details_data, colWidths=[2*inch, 4*inch])
        details_table.setStyle(TableStyle([
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONT', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2C3E50')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#34495E')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        elements.append(details_table)
        elements.append(Spacer(1, 0.5*inch))
        
        # Your interests
        if tourist['interests']:
            interests_text = Paragraph(
                f"<b>Your Interests:</b> {', '.join(tourist['interests'])}",
                self.styles['CustomBody']
            )
            elements.append(interests_text)
            elements.append(Spacer(1, 0.3*inch))
        
        # Generated date
        gen_date = Paragraph(
            f"<i>Generated on {datetime.now().strftime('%B %d, %Y')}</i>",
            self.styles['Normal']
        )
        elements.append(gen_date)
        
        return elements
    
    def _create_itinerary_details(self, data: Dict[str, Any]) -> list:
        """Create detailed daily itinerary"""
        elements = []
        
        # Section title
        title = Paragraph("Daily Itinerary", self.styles['CustomSubtitle'])
        elements.append(title)
        elements.append(Spacer(1, 0.2*inch))
        
        # Daily schedule
        for day in data['itinerary']['daily_schedule']:
            # Day heading
            day_title = Paragraph(
                f"Day {day['day']} - {day['date']} | {day['city']}",
                self.styles['DayHeading']
            )
            elements.append(day_title)
            
            # Sites to visit
            sites_text = f"<b>Sites to Visit:</b> {', '.join(day['sites'])}"
            sites_para = Paragraph(sites_text, self.styles['CustomBody'])
            elements.append(sites_para)
            elements.append(Spacer(1, 0.1*inch))
            
            # Activities
            if day['activities']:
                activities_text = f"<b>Suggested Activities:</b> {', '.join(day['activities'])}"
                activities_para = Paragraph(activities_text, self.styles['CustomBody'])
                elements.append(activities_para)
                elements.append(Spacer(1, 0.1*inch))
            
            # Cost
            cost_text = f"<b>Estimated Cost:</b> ${day['estimated_cost_usd']:.2f}"
            cost_para = Paragraph(cost_text, self.styles['CustomBody'])
            elements.append(cost_para)
            
            # Notes
            if day['notes']:
                notes_text = f"<i>{day['notes']}</i>"
                notes_para = Paragraph(notes_text, self.styles['Normal'])
                elements.append(notes_para)
            
            elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _create_cost_breakdown(self, data: Dict[str, Any]) -> list:
        """Create cost breakdown table"""
        elements = []
        
        # Section title
        title = Paragraph("Cost Breakdown", self.styles['CustomSubtitle'])
        elements.append(title)
        elements.append(Spacer(1, 0.2*inch))
        
        # Cost table
        cost_data = [['Day', 'Location', 'Estimated Cost']]
        
        for day in data['itinerary']['daily_schedule']:
            cost_data.append([
                f"Day {day['day']}",
                day['city'],
                f"${day['estimated_cost_usd']:.2f}"
            ])
        
        # Total row
        cost_data.append([
            'TOTAL',
            '',
            f"${data['itinerary']['total_cost_usd']:.2f}"
        ])
        
        cost_table = Table(cost_data, colWidths=[1*inch, 3*inch, 2*inch])
        cost_table.setStyle(TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            
            # Body
            ('ALIGN', (0, 1), (-1, -2), 'LEFT'),
            ('ALIGN', (2, 1), (2, -1), 'RIGHT'),
            ('FONTNAME', (0, 1), (-1, -2), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -2), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor('#ECF0F1')]),
            
            # Total row
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#2C3E50')),
            ('TEXTCOLOR', (0, -1), (-1, -1), colors.whitesmoke),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, -1), (-1, -1), 12),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        elements.append(cost_table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _create_recommendations(self, data: Dict[str, Any]) -> list:
        """Create recommendations and tips"""
        elements = []
        
        if 'recommendations' not in data:
            return elements
        
        recs = data['recommendations']
        
        # Section title
        title = Paragraph("Travel Tips & Recommendations", self.styles['CustomSubtitle'])
        elements.append(title)
        elements.append(Spacer(1, 0.2*inch))
        
        # Best season
        if recs.get('best_season'):
            season_text = f"<b>Best Season to Visit:</b> {recs['best_season']}"
            season_para = Paragraph(season_text, self.styles['CustomBody'])
            elements.append(season_para)
            elements.append(Spacer(1, 0.1*inch))
        
        # Packing tips
        if recs.get('packing_tips'):
            packing_title = Paragraph("<b>Packing Essentials:</b>", self.styles['CustomBody'])
            elements.append(packing_title)
            
            for tip in recs['packing_tips']:
                tip_para = Paragraph(f"• {tip}", self.styles['Normal'])
                elements.append(tip_para)
            
            elements.append(Spacer(1, 0.2*inch))
        
        # Accessibility info
        if recs.get('accessibility_info'):
            access_title = Paragraph("<b>Accessibility Information:</b>", self.styles['CustomBody'])
            elements.append(access_title)
            
            access_info = recs['accessibility_info']
            for key, value in access_info.items():
                info_para = Paragraph(f"• {value}", self.styles['Normal'])
                elements.append(info_para)
        
        return elements


# Test the PDF generator
if __name__ == "__main__":
    from tourism_backend_engine import TourismBackendEngine, TouristProfile
    
    print("=" * 80)
    print("PDF ITINERARY GENERATOR TEST")
    print("=" * 80 + "\n")
    
    # Initialize backend
    engine = TourismBackendEngine(
        '/mnt/user-data/outputs/master_clean_tourism_dataset_v1.csv'
    )
    
    # Create tourist profile
    tourist = TouristProfile(
        age=35,
        interests=['Art', 'History', 'Culture'],
        accessibility_needs=False,
        preferred_duration=7,
        budget_preference='Mid-range',
        climate_preference='Temperate'
    )
    
    # Generate itinerary
    itinerary = engine.generate_itinerary(tourist)
    
    # Generate PDF
    if itinerary['status'] == 'success':
        pdf_gen = PDFItineraryGenerator()
        pdf_path = '/mnt/user-data/outputs/sample_itinerary.pdf'
        
        pdf_gen.generate_itinerary_pdf(itinerary, pdf_path)
        
        print("\n✅ PDF generation complete!")
        print(f"   File: {pdf_path}")
    else:
        print("❌ Failed to generate itinerary")
