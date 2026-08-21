"use client";

import React, { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { 
  Mic, Square, Sparkles, RefreshCw, AlertTriangle, 
  HelpCircle, Volume2, Database, ShieldAlert, Award, ArrowRight, Trash2
} from "lucide-react";

interface ChunkCitation {
  id: string;
  text: string;
  score: number;
  strategy: string;
  metadata: {
    title?: string;
    language?: string;
    source_url?: string;
    original_query?: string;
    original_text?: string;
  };
}

interface LatencyTrace {
  stt: number;
  retrieval: number;
  guardrails_pre: number;
  generation: number;
  guardrails_post: number;
  total: number;
}

interface GuardrailStatus {
  passed_pre: boolean;
  passed_post: boolean;
  is_safe: boolean;
  is_on_topic: boolean;
  is_grounded: boolean;
  refusal_reason?: string;
}

interface QueryResponse {
  query: string;
  answer: string;
  retrieved_chunks: ChunkCitation[];
  latency_trace: LatencyTrace;
  guardrail_status: GuardrailStatus;
  success: boolean;
  error_message?: string;
}

export default function HomePage() {
  // Mic recording states
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [micPermissionDenied, setMicPermissionDenied] = useState(false);
  const [networkError, setNetworkError] = useState(false);
  
  // Pipeline status & results
  const [currentStage, setCurrentStage] = useState<"idle" | "recording" | "transcribing" | "retrieving" | "guardrails" | "generating" | "completed" | "error">("idle");
  const [stageTimers, setStageTimers] = useState<Record<string, number>>({});
  const [manualQuery, setManualQuery] = useState("");
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [selectedChunk, setSelectedChunk] = useState<ChunkCitation | null>(null);
  const [localTranscript, setLocalTranscript] = useState("");
  const localTranscriptRef = useRef("");
  const recognitionRef = useRef<any>(null);

  // Asset validation states
  const [assetsStatus, setAssetsStatus] = useState({
    logoMark: true,
    goaHindi: true,
    hackerSponsor: true
  });

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const stageTimerRef = useRef<number>(0);
  const stageIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Track recording duration
  useEffect(() => {
    if (isRecording) {
      setRecordingTime(0);
      timerIntervalRef.current = setInterval(() => {
        setRecordingTime(prev => {
          if (prev >= 30) {
            stopRecording();
            return 30;
          }
          return prev + 1;
        });
      }, 1000);
    } else {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    }
    return () => {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    };
  }, [isRecording]);

  // Clean intervals
  useEffect(() => {
    return () => {
      if (stageIntervalRef.current) clearInterval(stageIntervalRef.current);
    };
  }, []);

  // API base URL helper (handles absolute deploys and relative dev)
  const getApiUrl = (endpoint: string) => {
    const host = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    return `${host}${endpoint}`;
  };

  // Start stage-by-stage visual elapsed timer
  const startStageTimer = (stage: string) => {
    if (stageIntervalRef.current) clearInterval(stageIntervalRef.current);
    stageTimerRef.current = 0;
    setStageTimers(prev => ({ ...prev, [stage]: 0 }));
    
    stageIntervalRef.current = setInterval(() => {
      stageTimerRef.current += 10;
      setStageTimers(prev => ({ ...prev, [stage]: stageTimerRef.current }));
    }, 10);
  };

  const stopStageTimer = (stage: string, finalMs?: number) => {
    if (stageIntervalRef.current) clearInterval(stageIntervalRef.current);
    if (finalMs !== undefined) {
      setStageTimers(prev => ({ ...prev, [stage]: finalMs }));
    }
  };

  const startRecording = async () => {
    setErrorText(null);
    setNetworkError(false);
    setResponse(null);
    setSelectedChunk(null);
    setLocalTranscript("");
    localTranscriptRef.current = "";
    audioChunksRef.current = [];
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setMicPermissionDenied(false);
      
      // Select appropriate MIME type
      let mimeType = "audio/webm";
      if (MediaRecorder.isTypeSupported("audio/webm")) {
        mimeType = "audio/webm";
      } else if (MediaRecorder.isTypeSupported("audio/mp4")) {
        mimeType = "audio/mp4"; // Safari fallback
      } else if (MediaRecorder.isTypeSupported("audio/ogg")) {
        mimeType = "audio/ogg";
      } else {
        mimeType = ""; // Browser default
      }
      
      const options = mimeType ? { mimeType } : undefined;
      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType || "audio/wav" });
        // Close stream tracks to release microphone
        stream.getTracks().forEach(track => track.stop());
        
        if (recognitionRef.current) {
          recognitionRef.current.stop();
        }
        
        if (audioBlob.size < 1000) {
          setErrorText("Recording was too short or empty. Please speak again.");
          setCurrentStage("idle");
          return;
        }
        
        // Wait briefly for SpeechRecognition to settle its final chunk
        await new Promise(resolve => setTimeout(resolve, 250));
        await submitQuery(audioBlob, undefined, localTranscriptRef.current);
      };

      // Start browser Web Speech API for real-time local captioning & fallback transcript
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = "en-IN"; // English (India) is perfect for bilingual/Hinglish speech
        
        recognition.onresult = (event: any) => {
          let transcriptText = "";
          for (let i = 0; i < event.results.length; ++i) {
            transcriptText += event.results[i][0].transcript;
          }
          localTranscriptRef.current = transcriptText.trim();
          setLocalTranscript(transcriptText.trim());
        };
        
        recognition.onerror = (event: any) => {
          console.warn("Browser SpeechRecognition error:", event.error);
        };
        
        recognition.start();
        recognitionRef.current = recognition;
      }
      
      mediaRecorder.start(250); // Get audio slice every 250ms
      setIsRecording(true);
      setCurrentStage("recording");
    } catch (err: any) {
      console.error("Microphone access error:", err);
      setMicPermissionDenied(true);
      setCurrentStage("idle");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    }
  };

  // Submit audio blob (or text query) to backend API
  const submitQuery = async (audioBlob?: Blob, textQueryOverride?: string, browserTranscript?: string) => {
    try {
      const formData = new FormData();
      
      if (audioBlob) {
        formData.append("file", audioBlob, "query_audio.webm");
        if (browserTranscript) {
          formData.append("browser_transcript", browserTranscript);
        }
        setCurrentStage("transcribing");
        startStageTimer("stt");
      } else if (textQueryOverride) {
        formData.append("text_query", textQueryOverride);
        setCurrentStage("retrieving");
        startStageTimer("retrieval");
      } else {
        return;
      }
      
      const requestStart = performance.now();
      const response = await fetch(getApiUrl("/api/query"), {
        method: "POST",
        body: formData,
        headers: {
          // Note: Do not set Content-Type header when uploading files; browser will automatically set boundary
        }
      });
      
      if (response.status === 429) {
        throw new Error("Rate limit exceeded! Max 10 requests per minute.");
      }
      
      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || "Server error processing query.");
      }
      
      const data: QueryResponse = await response.json();
      
      // If voice query, complete STT timer first
      if (audioBlob) {
        stopStageTimer("stt", data.latency_trace.stt);
        setCurrentStage("retrieving");
        startStageTimer("retrieval");
        
        // Simulating rapid sequential steps to show progress matching backend logs
        await new Promise(r => setTimeout(r, Math.max(50, data.latency_trace.retrieval)));
        stopStageTimer("retrieval", data.latency_trace.retrieval);
      } else {
        stopStageTimer("retrieval", data.latency_trace.retrieval);
      }
      
      setCurrentStage("guardrails");
      startStageTimer("guardrails_pre");
      await new Promise(r => setTimeout(r, Math.max(20, data.latency_trace.guardrails_pre)));
      stopStageTimer("guardrails_pre", data.latency_trace.guardrails_pre);
      
      if (data.guardrail_status.passed_pre) {
        setCurrentStage("generating");
        startStageTimer("generation");
        await new Promise(r => setTimeout(r, Math.max(50, data.latency_trace.generation)));
        stopStageTimer("generation", data.latency_trace.generation);
        
        setCurrentStage("guardrails");
        startStageTimer("guardrails_post");
        await new Promise(r => setTimeout(r, Math.max(20, data.latency_trace.guardrails_post)));
        stopStageTimer("guardrails_post", data.latency_trace.guardrails_post);
      }
      
      setResponse(data);
      setCurrentStage("completed");
    } catch (err: any) {
      console.error("Query submission failed:", err);
      setErrorText(err.message || "Failed to contact backend API. Check if server is running.");
      setNetworkError(true);
      setCurrentStage("error");
    }
  };

  const handleManualSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!manualQuery.trim()) return;
    
    setErrorText(null);
    setNetworkError(false);
    setResponse(null);
    setSelectedChunk(null);
    
    const query = manualQuery;
    setManualQuery("");
    submitQuery(undefined, query);
  };

  // Helper to parse text citations and make them clickable
  const renderFormattedAnswer = (text: string) => {
    if (!text) return null;
    
    // Pattern to capture [Source X], 【Source X】, [source X], 【source1】, [src X] case-insensitively
    const parts = text.split(/([\[【](?:Source|Src)\s*:?\s*\d+[\]】])/gi);
    
    return parts.map((part, i) => {
      const match = part.match(/[\[【](?:Source|Src)\s*:?\s*(\d+)[\]】]/i);
      if (match) {
        const index = parseInt(match[1]) - 1;
        const chunk = response?.retrieved_chunks[index];
        
        return (
          <button
            key={i}
            onClick={() => chunk && setSelectedChunk(chunk)}
            className="inline-flex items-center px-1.5 py-0.5 mx-0.5 text-xs font-mono font-bold bg-pink hover:bg-green text-white transition-colors duration-150 rounded border border-charcoal cursor-pointer"
            title={chunk ? `View passage from ${chunk.metadata.title || 'Source'}` : "View Citation"}
          >
            ✦ Src {match[1]}
          </button>
        );
      }
      return <span key={i} className="font-serif text-lg leading-relaxed text-charcoal">{part}</span>;
    });
  };

  return (
    <div className="min-h-screen flex flex-col justify-between pt-24 pb-6 px-4 md:px-8 max-w-5xl mx-auto">
      
      {/* Top Floating Navigation Bar */}
      <nav className="fixed top-0 left-0 w-full z-50 bg-cream/95 border-b border-charcoal py-3 px-4 md:px-8 shadow-retro-small backdrop-blur-sm">
        <div className="max-w-5xl mx-auto flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            {assetsStatus.logoMark ? (
              <img 
                src="/assets/logo-mark.svg" 
                alt="HH Goa logo" 
                className="h-8 w-8 object-contain cursor-pointer hover:rotate-12 transition-transform"
                onClick={() => window.location.href = "/"}
              />
            ) : (
              <div 
                className="h-8 w-8 border border-dashed border-pink text-pink flex items-center justify-center font-bold text-xs bg-pink/10 cursor-pointer"
                onClick={() => window.location.href = "/"}
              >
                2:47
              </div>
            )}
            <div className="flex flex-col text-left">
              <span className="text-[9px] uppercase tracking-wider text-pink font-extrabold leading-none">HH Goa 2026</span>
              <h1 className="text-xs font-bold uppercase tracking-wider text-charcoal leading-tight">Voice RAG Console</h1>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Redirect to main hhgoa.com homepage */}
            <a 
              href="https://hhgoa.com/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="px-2.5 py-1.5 border border-charcoal bg-yellow text-charcoal hover:bg-green hover:text-cream transition-all text-[10px] font-bold uppercase tracking-wider flex items-center gap-1 shadow-retro-small"
            >
              HH Goa Site ↗
            </a>
            
            {/* Goa Hindi Wordmark */}
            {assetsStatus.goaHindi ? (
              <img 
                src="/assets/goa-hindi.svg" 
                alt="Goa Hindi wordmark" 
                className="h-7 object-contain hidden sm:block filter invert opacity-90"
                onError={() => setAssetsStatus(prev => ({ ...prev, goaHindi: false }))} 
              />
            ) : (
              <span className="hidden sm:inline border border-dashed border-green px-1.5 py-0.5 text-green text-[10px] font-bold bg-green/10">
                गोआ
              </span>
            )}

            <Link 
              href="/benchmark" 
              className="px-2.5 py-1.5 border border-charcoal bg-green text-cream hover:bg-pink hover:text-white transition-all text-[10px] font-bold uppercase tracking-wider shadow-retro-small"
            >
              /Benchmark
            </Link>
          </div>
        </div>
      </nav>

      {/* Warning if assets are missing */}
      {(!assetsStatus.logoMark || !assetsStatus.goaHindi || !assetsStatus.hackerSponsor) && (
        <div className="my-2 p-3 bg-red-100 border-2 border-pink text-charcoal text-xs font-mono flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-pink shrink-0" />
          <span>
            <strong>WARNING:</strong> Missing branding asset files in workspace! Please locate them inside the task folder and deploy inside <code>frontend/public/assets/</code>. Using dynamic fallback boxes instead.
          </span>
        </div>
      )}

      {/* 2. Main Content Zine Poster Layout */}
      <section className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-8 my-6">
        
        {/* Left Side: Recording panel (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          <div className="zine-border p-6 bg-cream shadow-retro flex flex-col justify-between h-full min-h-[380px] relative overflow-hidden">
            <div>
              <div className="flex items-center justify-between pb-4 zine-divider">
                <span className="text-xs uppercase tracking-widest font-bold text-pink">✦ Voice Input Capture</span>
                <span className="text-xs font-bold text-green">MAX: 30S</span>
              </div>
              
              <div className="py-6 text-center flex flex-col items-center justify-center">
                {isRecording ? (
                  <button 
                    onClick={stopRecording}
                    className="h-28 w-28 rounded-full border-2 border-charcoal bg-pink text-white flex items-center justify-center hover:scale-105 active:scale-95 transition-all shadow-retro-green animate-pulse"
                    aria-label="Stop Recording"
                  >
                    <Square className="h-10 w-10 fill-white" />
                  </button>
                ) : (
                  <button 
                    onClick={startRecording}
                    disabled={currentStage !== "idle" && currentStage !== "completed" && currentStage !== "error"}
                    className={`h-28 w-28 rounded-full border-2 border-charcoal flex items-center justify-center transition-all ${
                      currentStage !== "idle" && currentStage !== "completed" && currentStage !== "error"
                        ? "bg-slate/30 text-charcoal/30 cursor-not-allowed border-dashed"
                        : "bg-green text-cream hover:bg-pink hover:text-white shadow-retro cursor-pointer hover:scale-105 active:scale-95"
                    }`}
                    aria-label="Start Recording"
                  >
                    <Mic className="h-10 w-10" />
                  </button>
                )}

                <div className="mt-4">
                  {isRecording ? (
                    <div className="flex flex-col items-center">
                      <span className="text-pink font-bold text-xl animate-bounce">RECORDING...</span>
                      <span className="text-charcoal font-bold mt-1 text-2xl font-mono">00:{recordingTime.toString().padStart(2, '0')}</span>
                    </div>
                  ) : (
                    <div className="text-xs uppercase font-bold tracking-widest text-slate mt-2">
                      {currentStage === "idle" && "Tap microphone to speak question"}
                      {currentStage === "completed" && "Tap microphone to record new query"}
                      {currentStage === "error" && "An error occurred. Tap to retry"}
                      {currentStage !== "idle" && currentStage !== "completed" && currentStage !== "error" && "Pipeline is processing..."}
                    </div>
                  )}
                </div>

                {/* Live Captioning Display */}
                {isRecording && localTranscript && (
                  <div className="mt-4 p-3 border border-dashed border-pink/30 bg-pink/5 text-charcoal text-center text-xs font-mono rounded max-h-20 overflow-y-auto leading-relaxed animate-pulse">
                    &ldquo;{localTranscript}&rdquo;
                  </div>
                )}
              </div>
            </div>

            {/* In-app Instructions & Error handlers */}
            <div>
              {micPermissionDenied && (
                <div className="p-3 border border-pink bg-pink/10 text-xs text-charcoal mb-4 flex items-start gap-2">
                  <ShieldAlert className="h-4 w-4 text-pink shrink-0 mt-0.5" />
                  <div>
                    <span className="font-bold text-pink uppercase block">Mic Permission Denied</span>
                    Enable microphone permissions in your browser's settings bar to speak questions.
                  </div>
                </div>
              )}

              {networkError && (
                <div className="p-3 border border-pink bg-pink/10 text-xs text-charcoal mb-4">
                  <span className="font-bold text-pink uppercase block">Network Interruption</span>
                  Failed to connect to backend server. Double check that the FastAPI server is running.
                  <button 
                    onClick={() => submitQuery(undefined, "Verify connection")} 
                    className="mt-2 text-xs uppercase font-bold text-green underline block"
                  >
                    ✦ Retry connection check
                  </button>
                </div>
              )}

              {errorText && !networkError && (
                <div className="p-3 border border-pink bg-pink/10 text-xs text-charcoal mb-4 flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-pink shrink-0" />
                  <span>{errorText}</span>
                </div>
              )}

              {/* Text Query Form Fallback */}
              <form onSubmit={handleManualSubmit} className="zine-divider pt-4 flex gap-2">
                <input 
                  type="text" 
                  value={manualQuery}
                  onChange={(e) => setManualQuery(e.target.value)}
                  placeholder="Type question instead..."
                  disabled={isRecording}
                  className="flex-1 px-3 py-2 text-xs border border-charcoal bg-white focus:outline-none focus:ring-1 focus:ring-pink"
                />
                <button 
                  type="submit"
                  disabled={isRecording || !manualQuery.trim()}
                  className="px-3 py-2 border border-charcoal bg-charcoal text-cream font-bold text-xs uppercase hover:bg-pink hover:text-white transition-colors"
                >
                  <ArrowRight className="h-4 w-4" />
                </button>
              </form>
            </div>
          </div>
        </div>

        {/* Right Side: Process tracing and Answer (7 cols) */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          {/* A. Live pipeline tracer */}
          <div className="zine-border p-4 bg-cream">
            <div className="text-xs uppercase tracking-widest font-bold text-pink pb-2 zine-divider flex justify-between">
              <span>✦ Live Request Execution Trace</span>
              {response && (
                <span className="text-green font-bold">
                  TOTAL PIPELINE LATENCY: {response.latency_trace.total.toFixed(1)}ms
                </span>
              )}
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
              {[
                { label: "1. Speech-To-Text", id: "stt", activeStage: ["transcribing"] },
                { label: "2. Hybrid Retrieval", id: "retrieval", activeStage: ["retrieving"] },
                { label: "3. Pre-Guardrails", id: "guardrails_pre", activeStage: ["guardrails"] },
                { label: "4. LLM Generation", id: "generation", activeStage: ["generating"] },
              ].map((stage, idx) => {
                const isActive = stage.activeStage.includes(currentStage);
                const isFinished = stageTimers[stage.id] !== undefined;
                const elapsedMs = stageTimers[stage.id] || 0;
                
                return (
                  <div 
                    key={idx} 
                    className={`p-2 border text-left flex flex-col justify-between h-16 transition-colors ${
                      isActive 
                        ? "border-pink bg-pink/5" 
                        : isFinished 
                        ? "border-green bg-green/5" 
                        : "border-border-tan opacity-65"
                    }`}
                  >
                    <span className="text-[10px] uppercase font-bold text-slate tracking-tight">
                      {stage.label}
                    </span>
                    <span className={`text-xs font-mono font-bold ${isActive ? 'text-pink' : isFinished ? 'text-green' : 'text-slate'}`}>
                      {isActive ? "Running..." : isFinished ? `✓ ${elapsedMs.toFixed(1)}ms` : "Queued"}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* B. Grounded Answer Panel */}
          <div className="zine-border p-6 bg-cream shadow-retro flex-1 flex flex-col justify-between min-h-[350px]">
            <div>
              <div className="flex items-center justify-between pb-3 zine-divider">
                <span className="text-xs uppercase tracking-widest font-bold text-pink">✦ Grounded Answer Output</span>
                {response && (
                  <div className="flex gap-2">
                    {response.guardrail_status.is_safe ? (
                      <span className="px-1.5 py-0.5 bg-green/10 text-green border border-green text-[10px] font-bold">SAFE</span>
                    ) : (
                      <span className="px-1.5 py-0.5 bg-pink/10 text-pink border border-pink text-[10px] font-bold">UNSAFE</span>
                    )}
                    {response.guardrail_status.is_on_topic ? (
                      <span className="px-1.5 py-0.5 bg-green/10 text-green border border-green text-[10px] font-bold">ON-DOMAIN</span>
                    ) : (
                      <span className="px-1.5 py-0.5 bg-pink/10 text-pink border border-pink text-[10px] font-bold">OUT-OF-DOMAIN</span>
                    )}
                  </div>
                )}
              </div>

              {response ? (
                <div className="py-4">
                  {/* User query transcription review */}
                  <div className="mb-4 bg-white/50 p-3 border border-border-tan text-xs">
                    <span className="font-bold text-slate block mb-1 font-mono uppercase tracking-widest">Transcribed Query:</span>
                    <p className="font-serif italic text-charcoal text-base">"{response.query}"</p>
                  </div>
                  
                  {/* Answer presentation */}
                  <div className="mt-2 text-charcoal font-serif">
                    {renderFormattedAnswer(response.answer)}
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-20 text-slate text-xs font-mono">
                  {currentStage === "idle" && (
                    <>
                      <Sparkles className="h-10 w-10 text-pink/50 mb-2 stroke-[1.5]" />
                      <span>WAITING FOR VOICE QUERY INPUT</span>
                    </>
                  )}
                  {currentStage === "recording" && (
                    <>
                      <Mic className="h-10 w-10 text-pink mb-2 animate-bounce" />
                      <span className="text-pink font-bold">CAPTURING AUDIO INPUT...</span>
                    </>
                  )}
                  {currentStage !== "idle" && currentStage !== "recording" && currentStage !== "completed" && currentStage !== "error" && (
                    <div className="flex flex-col items-center">
                      <RefreshCw className="h-10 w-10 text-green animate-spin mb-2" />
                      <span className="text-green font-bold uppercase">Executing Pipeline stages...</span>
                    </div>
                  )}
                  {currentStage === "error" && (
                    <>
                      <AlertTriangle className="h-10 w-10 text-pink mb-2" />
                      <span className="text-pink font-bold uppercase">Execution halted with error</span>
                    </>
                  )}
                </div>
              )}
            </div>

            {/* Citations panel details inside answer box */}
            {response && response.retrieved_chunks.length > 0 && (
              <div className="zine-divider pt-4 mt-auto">
                <span className="text-[10px] uppercase font-bold tracking-widest text-slate block mb-2">
                  ✦ Click Source Chunks used for Grounding:
                </span>
                <div className="flex flex-wrap gap-2">
                  {response.retrieved_chunks.map((chunk, idx) => (
                    <button
                      key={idx}
                      onClick={() => setSelectedChunk(chunk)}
                      className={`px-3 py-1 text-xs border font-mono transition-all ${
                        selectedChunk?.id === chunk.id 
                          ? "bg-green text-cream border-charcoal scale-105" 
                          : "bg-white hover:bg-border-tan text-charcoal border-border-tan"
                      }`}
                    >
                      Source {idx + 1} ({chunk.strategy})
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* 3. Interactive Citations Panel (Reveals on source click) */}
      {selectedChunk && (
        <section className="my-4 p-4 zine-border bg-white shadow-retro animate-fadeIn">
          <div className="flex items-center justify-between pb-2 border-b border-border-tan">
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4 text-green" />
              <span className="text-xs font-bold uppercase text-charcoal tracking-wide">
                Grounded Source Passage Analysis ({selectedChunk.strategy} chunker)
              </span>
            </div>
            <button 
              onClick={() => setSelectedChunk(null)} 
              className="text-xs font-bold text-pink uppercase underline hover:text-green cursor-pointer"
            >
              Close Drawer [X]
            </button>
          </div>
          <div className="pt-3 grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="md:col-span-3">
              <span className="text-[10px] font-bold text-slate block uppercase tracking-wider mb-1">Passage Segment Content:</span>
              <p className="font-serif text-sm leading-relaxed text-charcoal bg-cream/35 p-3 border border-border-tan font-light">
                {selectedChunk.metadata.original_text || selectedChunk.text}
              </p>
            </div>
            <div className="text-xs font-mono">
              <span className="text-[10px] font-bold text-slate block uppercase tracking-wider mb-1">Source Details:</span>
              <ul className="space-y-1.5">
                <li><strong>Title:</strong> {selectedChunk.metadata.title || "MSMARCO Document"}</li>
                <li><strong>Lang:</strong> {selectedChunk.metadata.language?.toUpperCase() || "HI"}</li>
                <li><strong>Score:</strong> {selectedChunk.score.toFixed(4)}</li>
                <li><strong>Link:</strong> <a href={selectedChunk.metadata.source_url || "#"} target="_blank" className="text-pink underline break-all">Source Link ↗</a></li>
              </ul>
            </div>
          </div>
        </section>
      )}

      {/* 4. Footer */}
      <footer className="w-full mt-10 pt-6 zine-divider flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="flex items-center gap-3 p-2 border border-transparent hover:border-charcoal hover:bg-yellow hover:shadow-retro-small transition-all duration-300 cursor-pointer">
          {assetsStatus.hackerSponsor ? (
            <img 
              src="/assets/hacker-house-sponsor.png" 
              alt="Hacker House logo" 
              className="h-10 object-contain hover:scale-105 transition-transform duration-300"
              onError={() => setAssetsStatus(prev => ({ ...prev, hackerSponsor: false }))} 
            />
          ) : (
            <div className="border border-dashed border-slate px-2 py-1 text-slate text-xs font-bold bg-slate/5" title="Missing hacker-house-sponsor.png">
              Hacker House Sponsor
            </div>
          )}
          <div className="flex flex-col text-left">
            <span className="text-xs text-charcoal font-bold uppercase tracking-wider">Hacker House Organizer</span>
            <span className="text-[10px] text-slate font-mono">India's Biggest Build residency</span>
          </div>
        </div>

        <div className="text-center md:text-right">
          <p className="text-xs font-bold uppercase tracking-widest text-pink mb-1">#RAGInGoa</p>
          <p className="text-[10px] text-slate font-mono">
            Built for #RAGInGoa — HH Goa 2026, Task #2 · © 2026 HH-Goa.
          </p>
        </div>
      </footer>
    </div>
  );
}
