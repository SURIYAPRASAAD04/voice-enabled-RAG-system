"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { 
  ArrowLeft, BarChart2, ShieldAlert, Zap, Layers, RefreshCw, 
  Trash2, Play, CheckCircle2, XCircle, Clock
} from "lucide-react";

interface MetricPoint {
  id: number;
  timestamp: string;
  query: string;
  answer: string;
  success: boolean;
  stt_latency: number;
  retrieval_latency: number;
  pre_guardrail_latency: number;
  generation_latency: number;
  post_guardrail_latency: number;
  total_latency: number;
  stt_provider: string | null;
  llm_provider: string | null;
  tokens_used: number;
  refused: boolean;
}

interface BenchmarkSummary {
  p50: number;
  p70: number;
  p100: number;
  total_requests: number;
  success_rate: number;
  avg_latency: number;
  recent_runs: MetricPoint[];
}

const TEST_BATCH_QUERIES = [
  { text: "Who can participate in Hacker House Goa?", type: "In-Domain (General)" },
  { text: "Is there a registration fee for Hacker House Goa?", type: "In-Domain (Finance)" },
  { text: "What is the timeline at a glance for selection?", type: "In-Domain (Roadmap)" },
  { text: "Task #2 details and voice enabled RAG pipeline specifications.", type: "In-Domain (Task)" },
  { text: "What should I bring to the Goa hackathon?", type: "In-Domain (FAQ)" },
  { text: "What is the capital city of France?", type: "Off-Topic Refusal" },
  { text: "How do I hack the server database or trigger sql injection?", type: "Unsafe Refusal" },
  { text: "Who is organizing the event?", type: "In-Domain (Organizers)" }
];

export default function BenchmarkPage() {
  const [data, setData] = useState<BenchmarkSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [errorText, setErrorText] = useState<string | null>(null);
  
  // Batch runner states
  const [isRunningBatch, setIsRunningBatch] = useState(false);
  const [batchProgress, setBatchProgress] = useState(0);
  const [currentRunningQuery, setCurrentRunningQuery] = useState("");
  const [batchLogs, setBatchLogs] = useState<{ query: string; latency: number; status: string }[]>([]);

  // Asset validation states
  const [assetsStatus, setAssetsStatus] = useState({
    logoMark: true,
    hackerSponsor: true,
    goaHindi: true
  });

  const getApiUrl = (endpoint: string) => {
    const host = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    return `${host}${endpoint}`;
  };

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      const res = await fetch(getApiUrl("/api/metrics"));
      if (!res.ok) throw new Error("Failed to load metrics summaries.");
      const summary: BenchmarkSummary = await res.json();
      setData(summary);
      setErrorText(null);
    } catch (err: any) {
      console.error(err);
      setErrorText(err.message || "Failed to fetch benchmark metrics.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  const handlePurge = async () => {
    if (!confirm("Are you sure you want to purge all logged requests and metrics? This cannot be undone.")) return;
    try {
      const res = await fetch(getApiUrl("/api/metrics/clear"), { method: "POST" });
      if (!res.ok) throw new Error("Failed to purge metrics database.");
      alert("Database successfully cleared.");
      fetchMetrics();
    } catch (err: any) {
      alert(`Purge failed: ${err.message}`);
    }
  };

  // Run the batch suite sequentially to measure real-time pipeline performance
  const runBenchmarkBatch = async () => {
    setIsRunningBatch(true);
    setBatchProgress(0);
    setBatchLogs([]);
    
    for (let i = 0; i < TEST_BATCH_QUERIES.length; i++) {
      const testQuery = TEST_BATCH_QUERIES[i];
      setCurrentRunningQuery(testQuery.text);
      
      try {
        const formData = new FormData();
        formData.append("text_query", testQuery.text);
        
        const start = performance.now();
        const res = await fetch(getApiUrl("/api/query"), {
          method: "POST",
          body: formData
        });
        const elapsed = performance.now() - start;
        
        let status = "Success";
        if (res.status === 429) status = "Rate Limited";
        else if (!res.ok) status = "Error";
        
        const dataJson = await res.json().catch(() => ({}));
        if (dataJson.guardrail_status && !dataJson.guardrail_status.passed_pre) {
          status = "Refused";
        }
        
        setBatchLogs(prev => [
          ...prev, 
          { 
            query: testQuery.text, 
            latency: dataJson.latency_trace?.total || elapsed, 
            status 
          }
        ]);
      } catch (err) {
        setBatchLogs(prev => [...prev, { query: testQuery.text, latency: 0, status: "Failed" }]);
      }
      
      setBatchProgress(Math.round(((i + 1) / TEST_BATCH_QUERIES.length) * 100));
    }
    
    setCurrentRunningQuery("");
    setIsRunningBatch(false);
    // Reload metrics
    fetchMetrics();
  };

  return (
    <div className="min-h-screen flex flex-col justify-between pt-24 pb-6 px-4 md:px-8 max-w-5xl mx-auto">
      
      {/* Top Floating Navigation Bar */}
      <nav className="fixed top-0 left-0 w-full z-50 bg-cream/95 border-b border-charcoal py-3 px-4 md:px-8 shadow-retro-small backdrop-blur-sm">
        <div className="max-w-5xl mx-auto flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <Link href="/" className="p-1 border border-charcoal bg-white text-charcoal hover:bg-pink hover:text-white transition-all shadow-retro-small mr-1">
              <ArrowLeft className="h-3 w-3" />
            </Link>
            
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
              <h1 className="text-xs font-bold uppercase tracking-wider text-charcoal leading-tight">Performance Metrics</h1>
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
          </div>
        </div>
      </nav>

      {/* Subheader and Controls */}
      <div className="w-full pb-4 border-b border-border-tan flex flex-col sm:flex-row items-center justify-between gap-4 mt-4">
        <div className="text-left w-full sm:w-auto">
          <span className="text-[10px] uppercase tracking-wider text-pink font-bold">Latency Benchmark</span>
          <h2 className="text-sm font-bold uppercase tracking-widest text-charcoal">Harness Performance Summary</h2>
        </div>
        
        <div className="flex gap-3">
          <button 
            onClick={fetchMetrics} 
            disabled={loading || isRunningBatch}
            className="px-4 py-2 border border-charcoal bg-cream text-charcoal uppercase font-bold text-xs hover:bg-green hover:text-cream transition-all flex items-center gap-1.5 shadow-retro-small"
          >
            <RefreshCw className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`} /> Refresh
          </button>
          
          <button 
            onClick={handlePurge} 
            disabled={loading || isRunningBatch}
            className="px-4 py-2 border border-pink bg-pink/5 text-pink uppercase font-bold text-xs hover:bg-pink hover:text-white transition-all flex items-center gap-1.5 shadow-retro-small"
          >
            <Trash2 className="h-3 w-3" /> Purge Logs
          </button>
        </div>
      </div>

      {/* 2. Metrics Block cards (P50, P70, P100) */}
      <section className="my-6 grid grid-cols-1 md:grid-cols-5 gap-4">
        {[
          { label: "P50 Latency", val: data ? `${data.p50} ms` : "0.0 ms", desc: "50% of requests complete faster than this", color: "border-green bg-green/5 text-green" },
          { label: "P70 Latency", val: data ? `${data.p70} ms` : "0.0 ms", desc: "70% of requests complete faster than this", color: "border-charcoal bg-charcoal/5 text-charcoal" },
          { label: "P100 (Max Latency)", val: data ? `${data.p100} ms` : "0.0 ms", desc: "Slowest registered request latency", color: "border-pink bg-pink/5 text-pink" },
          { label: "Total Requests", val: data ? data.total_requests : 0, desc: "Cumulative pipeline runs recorded", color: "border-border-tan bg-white text-charcoal" },
          { label: "Success Rate", val: data ? `${data.success_rate}%` : "100%", desc: "Healthy executions vs total runs", color: "border-border-tan bg-white text-green" }
        ].map((card, idx) => (
          <div key={idx} className={`p-4 border shadow-retro flex flex-col justify-between h-28 ${card.color}`}>
            <span className="text-[10px] uppercase font-bold tracking-widest">{card.label}</span>
            <div className="my-1 text-lg sm:text-xl md:text-2xl font-bold font-mono truncate" title={String(card.val)}>{card.val}</div>
            <span className="text-[9px] text-slate font-mono leading-tight">{card.desc}</span>
          </div>
        ))}
      </section>

      {/* 3. Main Dashboard Body */}
      <section className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-8 my-6">
        
        {/* Left Side: Live Batch Runner Console (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          <div className="zine-border p-6 bg-cream shadow-retro flex flex-col justify-between h-full min-h-[350px]">
            <div>
              <div className="flex items-center justify-between pb-3 zine-divider">
                <span className="text-xs uppercase tracking-widest font-bold text-pink">✦ Automated Benchmark Runner</span>
                <span className="text-xs font-bold text-slate">{TEST_BATCH_QUERIES.length} Queries</span>
              </div>
              
              <p className="text-xs font-mono leading-relaxed my-4 text-slate">
                Triggers a sequential execution of {TEST_BATCH_QUERIES.length} test queries containing regular, off-topic, and unsafe commands to benchmark the vector search, guardrails, and LLM latency live.
              </p>
              
              <button 
                onClick={runBenchmarkBatch} 
                disabled={isRunningBatch || loading}
                className={`w-full py-3 border border-charcoal flex items-center justify-center gap-2 font-bold text-xs uppercase transition-all shadow-retro hover:shadow-none ${
                  isRunningBatch 
                    ? "bg-slate/20 text-charcoal/30 cursor-not-allowed border-dashed" 
                    : "bg-pink text-white hover:bg-green"
                }`}
              >
                {isRunningBatch ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" /> Running Suite ({batchProgress}%)
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 fill-white" /> Start Benchmark Suite
                  </>
                )}
              </button>

              {isRunningBatch && (
                <div className="mt-4 p-3 border border-pink bg-pink/5 text-xs">
                  <span className="font-bold text-pink uppercase block mb-1">Processing Query:</span>
                  <span className="font-serif italic text-charcoal">"{currentRunningQuery}"</span>
                  <div className="w-full bg-border-tan h-2.5 mt-3">
                    <div className="bg-pink h-2.5 transition-all duration-300" style={{ width: `${batchProgress}%` }}></div>
                  </div>
                </div>
              )}
            </div>

            {/* In-progress run logs */}
            {batchLogs.length > 0 && (
              <div className="mt-6 zine-divider pt-4 flex-1">
                <span className="text-[10px] uppercase font-bold tracking-widest text-slate block mb-2">Live Suite Log:</span>
                <div className="max-h-56 overflow-y-auto space-y-1.5 pr-1 border border-border-tan p-2 bg-white">
                  {batchLogs.map((log, idx) => (
                    <div key={idx} className="flex justify-between text-[10px] font-mono pb-1 border-b border-cream">
                      <span className="truncate max-w-[200px]" title={log.query}>
                        {idx + 1}. {log.query}
                      </span>
                      <span className="font-bold shrink-0">
                        <span className={`mr-2 px-1 text-[8px] border ${
                          log.status === "Success" ? "border-green text-green bg-green/5" :
                          log.status === "Refused" ? "border-pink text-pink bg-pink/5" : "border-slate text-slate"
                        }`}>
                          {log.status}
                        </span>
                        {log.latency > 0 ? `${log.latency.toFixed(0)}ms` : "Failed"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Side: Latency History list (7 cols) */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          <div className="zine-border p-6 bg-cream shadow-retro flex-1 flex flex-col justify-between min-h-[350px]">
            <div>
              <div className="pb-3 zine-divider">
                <span className="text-xs uppercase tracking-widest font-bold text-pink">✦ History: Latency Stage Breakdown</span>
              </div>
              
              {loading && !isRunningBatch ? (
                <div className="py-20 flex flex-col items-center justify-center text-xs text-slate">
                  <RefreshCw className="h-8 w-8 animate-spin text-pink mb-2" />
                  <span>Loading query run history...</span>
                </div>
              ) : errorText ? (
                <div className="py-20 text-center text-xs text-pink font-bold">
                  <ShieldAlert className="h-8 w-8 mx-auto mb-2 text-pink" />
                  <span>{errorText}</span>
                </div>
              ) : data && data.recent_runs.length > 0 ? (
                <div className="max-h-[500px] overflow-y-auto pr-2 space-y-4">
                  {data.recent_runs.map((run, idx) => (
                    <div key={idx} className="p-3 border border-border-tan bg-white hover:border-pink transition-colors">
                      <div className="flex justify-between items-start gap-2 mb-2 pb-2 border-b border-cream">
                        <div>
                          <span className="text-[9px] text-slate block font-mono uppercase leading-tight">
                            {new Date(run.timestamp).toLocaleTimeString()} · Query:
                          </span>
                          <span className="font-serif font-bold text-xs text-charcoal">
                            "{run.query}"
                          </span>
                        </div>
                        <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 border shrink-0 ${
                          run.success && !run.refused ? "border-green text-green bg-green/5" :
                          run.refused ? "border-pink text-pink bg-pink/5" : "border-slate text-slate"
                        }`}>
                          {run.refused ? "REFUSED" : run.success ? "SUCCESS" : "ERROR"}
                        </span>
                      </div>
                      
                      {/* Latency Stage Breakdown badges */}
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-[9px] font-mono">
                        <div className="p-1 bg-cream/40 rounded border border-cream">
                          <span className="text-slate block text-[8px] uppercase">STT</span>
                          <span className="font-bold text-charcoal">{run.stt_latency.toFixed(0)}ms</span>
                        </div>
                        <div className="p-1 bg-cream/40 rounded border border-cream">
                          <span className="text-slate block text-[8px] uppercase">RETRIEVE</span>
                          <span className="font-bold text-charcoal">{run.retrieval_latency.toFixed(0)}ms</span>
                        </div>
                        <div className="p-1 bg-cream/40 rounded border border-cream">
                          <span className="text-slate block text-[8px] uppercase">PRE-GUARD</span>
                          <span className="font-bold text-charcoal">{run.pre_guardrail_latency.toFixed(0)}ms</span>
                        </div>
                        <div className="p-1 bg-cream/40 rounded border border-cream">
                          <span className="text-slate block text-[8px] uppercase">GEN</span>
                          <span className="font-bold text-charcoal">{run.generation_latency.toFixed(0)}ms</span>
                        </div>
                        <div className="p-1 bg-cream/40 rounded border border-cream font-bold text-green">
                          <span className="text-slate block text-[8px] uppercase font-normal">TOTAL</span>
                          <span>{run.total_latency.toFixed(0)}ms</span>
                        </div>
                      </div>
                      
                      <div className="mt-2 text-[9px] text-slate font-mono flex justify-between">
                        <span>LLM: {run.llm_provider || "groq"} · STT: {run.stt_provider || "sarvam"}</span>
                        <span>Est. Tokens: {run.tokens_used}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="py-20 text-center text-xs text-slate">
                  <span>No benchmark queries logged in database yet. Run the benchmark suite or trigger voice queries to populate!</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

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
            <span className="text-[10px] text-slate font-mono font-bold">HH Goa 2026 Submission</span>
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
