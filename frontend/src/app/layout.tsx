import type { Metadata } from "next";
import { Playfair_Display, Space_Mono } from "next/font/google";
import "./globals.css";

const playfair = Playfair_Display({
  variable: "--font-playfair",
  subsets: ["latin"],
  display: "swap",
});

const spaceMono = Space_Mono({
  variable: "--font-space-mono",
  subsets: ["latin"],
  weight: ["400", "700"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "HH Goa 2026 | Voice-Enabled RAG Model",
  description: "HH Goa 2026 Task #2 Submission: A voice-to-answer RAG pipeline with hybrid retrieval, guardrails, and sub-200ms orchestration.",
  keywords: ["Hacker House Goa", "HH Goa 2026", "RAG", "Voice RAG", "Sarvam AI", "Qdrant", "FastAPI"],
  authors: [{ name: "Hacker House Team" }],
  viewport: "width=device-width, initial-scale=1",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${playfair.variable} ${spaceMono.variable} h-full antialiased`}
    >
      <body className="min-h-full bg-cream text-charcoal font-mono selection:bg-pink selection:text-white flex flex-col relative">
        {/* Dynamic Paper Grain Overlay */}
        <div className="paper-grain" aria-hidden="true" />
        
        {/* Main Application Container */}
        <main className="flex-1 flex flex-col relative z-10">
          {children}
        </main>
      </body>
    </html>
  );
}
