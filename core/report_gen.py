from fpdf import FPDF
import pandas as pd
import base64
import tempfile

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, '🏎️ APEXBRAIN | POST-SESSION DEBRIEF', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, f'  {title}', 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

class ReportGenerator:
    def generate_pdf_report(self, session, driver_1, driver_2, strategy_insight, battle_insight):
        """
        Generates a 2-Page Engineering PDF.
        """
        pdf = PDFReport()
        pdf.add_page()
        
        # --- PAGE 1: SESSION SUMMARY ---
        pdf.chapter_title(f"SESSION: {session.event['EventName']} - {session.name}")
        
        # Weather
        w = session.weather_data.iloc[0]
        weather_txt = f"Track Temp: {w['TrackTemp']}C | Air Temp: {w['AirTemp']}C | Humidity: {w['Humidity']}%"
        pdf.chapter_body(weather_txt)
        
        # Results
        if hasattr(session, 'results') and not session.results.empty:
            winner = session.results.iloc[0]
            res_txt = f"WINNER: {winner['Abbreviation']} ({winner['TeamName']})\n"
            fastest = session.laps.pick_quicklaps().sort_values('LapTime').iloc[0]
            res_txt += f"FASTEST LAP: {fastest['Driver']} - {str(fastest['LapTime']).split()[-1][:-3]}"
            pdf.chapter_body(res_txt)
            
        pdf.ln(5)
        
        # --- PAGE 1: BATTLE ANALYSIS ---
        pdf.chapter_title(f"HEAD-TO-HEAD: {driver_1} vs {driver_2}")
        pdf.chapter_body("AI INSIGHT:")
        pdf.set_font('Arial', 'I', 10)
        pdf.multi_cell(0, 5, battle_insight)
        pdf.set_font('Arial', '', 10)
        pdf.ln(5)
        
        pdf.chapter_body(f"Note: Full telemetry trace available in the dashboard. {driver_1} comparisons against {driver_2} show variance in braking points and corner exit traction.")
        
        # --- PAGE 2: STRATEGY ---
        pdf.add_page()
        pdf.chapter_title("STRATEGY RECOMMENDATION")
        pdf.chapter_body(strategy_insight)
        
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 10, 'CONFIDENTIAL - INTERNAL USE ONLY', 0, 0, 'C')

        # Output to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(temp_file.name)
        return temp_file.name

    def get_download_link(self, file_path, filename="Debrief.pdf"):
        with open(file_path, "rb") as f:
            bytes_data = f.read()
        b64 = base64.b64encode(bytes_data).decode()
        return f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="background-color:#ef4444; color:white; padding:10px 20px; text-decoration:none; border-radius:5px; font-weight:bold;">📄 DOWNLOAD PDF REPORT</a>'