import React, { useState, useEffect, useRef } from 'react';
import { 
  FileText, 
  Youtube, 
  Image as ImageIcon, 
  Music, 
  Send, 
  Trash2, 
  RotateCcw, 
  AlertCircle, 
  CheckCircle, 
  Sparkles, 
  BookOpen, 
  Layers, 
  X,
  FileCheck,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Sun,
  Moon,
  Menu,
  Plus
} from 'lucide-react';

const API_BASE = window.location.origin.includes("localhost:5173") 
  ? "http://localhost:8000/api" 
  : "/api";

// --- Simple Regex Markdown Parser to support bold, headers, list and tables safely ---
const parseMarkdown = (markdown) => {
  if (!markdown) return "";
  
  let html = markdown;
  
  // Escape script tags
  html = html.replace(/<script[^>]*>([\s\S]*?)<\/script>/gi, '');
  
  // Table Parsing
  const lines = html.split('\n');
  let inTable = false;
  let tableRows = [];
  let tableHeader = true;
  let newLines = [];
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith('|') && line.endsWith('|')) {
      if (!inTable) {
        inTable = true;
        tableRows = [];
        tableHeader = true;
      }
      
      // Skip separator row like |---|---|
      if (line.includes('---') || line.includes('- -')) {
        tableHeader = false;
        continue;
      }
      
      const cells = line.split('|')
        .slice(1, -1)
        .map(cell => cell.trim());
      
      const cellTag = tableHeader ? 'th' : 'td';
      const rowContent = cells.map(cell => `<${cellTag}>${cell}</${cellTag}>`).join('');
      tableRows.push(`<tr>${rowContent}</tr>`);
      tableHeader = false;
    } else {
      if (inTable) {
        newLines.push(`<table><tbody>${tableRows.join('')}</tbody></table>`);
        inTable = false;
      }
      newLines.push(lines[i]);
    }
  }
  if (inTable) {
    newLines.push(`<table><tbody>${tableRows.join('')}</tbody></table>`);
  }
  
  html = newLines.join('\n');
  
  // Headers
  html = html.replace(/^### (.*?)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.*?)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.*?)$/gm, '<h1>$1</h1>');
  
  // Bold
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  
  // Unordered Lists
  html = html.replace(/^\- (.*?)$/gm, '<li>$1</li>');
  html = html.replace(/^\* (.*?)$/gm, '<li>$1</li>');
  
  // Wrap list items in <ul>. Clean up consecutive list items
  html = html.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');
  
  // Linebreaks
  html = html.replace(/\n/g, '<br/>');
  
  // Clean empty tags created by breaks
  html = html.replace(/<br\/><br\/>/g, '<br/>');
  
  return html;
};

export default function App() {
  // Session / ID
  const sessionId = "default";

  // Theme State (NotebookLM style)
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 'dark';
  });

  // Sidebar Visibility State (responsive hamburger toggle)
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth > 1024);

  // Ingestion Accordion open state (+ Add Source)
  const [ingestDrawerOpen, setIngestDrawerOpen] = useState(false);

  // Mobile Active Panel Selector ('chat' or 'guide')
  const [mobileActivePanel, setMobileActivePanel] = useState('chat');

  // Handle theme state changes on HTML element
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  // Adjust sidebar state automatically on window resizing
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth > 1024) {
        setSidebarOpen(true);
      } else {
        setSidebarOpen(false);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  // Tab State
  const [activeTab, setActiveTab] = useState('url'); // 'url', 'pdf', 'image', 'media'

  // Workspace Ingestion Files/Data State
  const [ytUrls, setYtUrls] = useState(['']);
  const [pdfFiles, setPdfFiles] = useState([]);
  const [imageFile, setImageFile] = useState(null);
  const [mediaFile, setMediaFile] = useState(null);
  
  // Status State
  const [workspaceStatus, setWorkspaceStatus] = useState({
    document_ids: [],
    has_active_image: false,
    active_image_name: "",
    message_count: 0,
    is_processing: false,
    processing_message: "",
    has_summary: false,
    has_quiz: false,
    has_comparison: false
  });

  // Chat State
  const [chatHistory, setChatHistory] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  // Analysis Panels State
  const [activeAnalysisMode, setActiveAnalysisMode] = useState(null); // null, 'summary', 'quiz', 'compare'
  const [summaryData, setSummaryData] = useState("");
  const [comparisonData, setComparisonData] = useState("");
  const [quizData, setQuizData] = useState(null);
  const [quizAnswers, setQuizAnswers] = useState({}); // {questionIndex: selectedOption}
  const [quizRevealed, setQuizRevealed] = useState({}); // {questionIndex: true/false}

  // File picker refs
  const pdfInputRef = useRef(null);
  const imgInputRef = useRef(null);
  const mediaInputRef = useRef(null);

  // Auto-scroll chat
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory, isTyping]);

  // Fetch status on startup
  useEffect(() => {
    fetchStatus();
    fetchChatHistory();
  }, []);

  // Poll status while backend is processing ingestion
  useEffect(() => {
    let interval;
    if (workspaceStatus.is_processing) {
      interval = setInterval(() => {
        fetchStatus();
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [workspaceStatus.is_processing]);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/status?session_id=${sessionId}`);
      if (res.ok) {
        const data = await res.json();
        setWorkspaceStatus(data);
      }
    } catch (e) {
      console.error("Failed to fetch status:", e);
    }
  };

  const fetchChatHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/chat/history?session_id=${sessionId}`);
      if (res.ok) {
        const data = await res.json();
        setChatHistory(data.messages);
      }
    } catch (e) {
      console.error("Failed to fetch chat history:", e);
    }
  };

  // --- Handlers for Ingestions ---

  const handleYtSubmit = async (e) => {
    e.preventDefault();
    const validUrls = ytUrls.map(u => u.trim()).filter(Boolean);
    if (validUrls.length === 0) return;

    setWorkspaceStatus(prev => ({
      ...prev,
      is_processing: true,
      processing_message: "Triggering YouTube transcription pipeline..."
    }));

    try {
      const res = await fetch(`${API_BASE}/process/youtube`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ urls: validUrls.join(','), session_id: sessionId })
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "YouTube processing failed");
      
      setYtUrls(['']);
      alert(data.message || "YouTube content successfully indexed!");
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      fetchStatus();
    }
  };

  const handlePdfSubmit = async () => {
    if (pdfFiles.length === 0) return;

    setWorkspaceStatus(prev => ({
      ...prev,
      is_processing: true,
      processing_message: "Processing and indexing PDF elements..."
    }));

    const formData = new FormData();
    for (let i = 0; i < pdfFiles.length; i++) {
      formData.append('files', pdfFiles[i]);
    }
    formData.append('session_id', sessionId);

    try {
      const res = await fetch(`${API_BASE}/upload/pdf`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "PDF processing failed");

      setPdfFiles([]);
      if (pdfInputRef.current) pdfInputRef.current.value = "";
      alert(data.message || "PDFs successfully indexed!");
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      fetchStatus();
    }
  };

  const handleImageSubmit = async () => {
    if (!imageFile) return;

    setWorkspaceStatus(prev => ({
      ...prev,
      is_processing: true,
      processing_message: "Uploading visual image content..."
    }));

    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('session_id', sessionId);

    try {
      const res = await fetch(`${API_BASE}/upload/image`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Image processing failed");

      setImageFile(null);
      if (imgInputRef.current) imgInputRef.current.value = "";
      alert(data.message || "Image successfully uploaded!");
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      fetchStatus();
    }
  };

  const handleMediaSubmit = async () => {
    if (!mediaFile) return;

    setWorkspaceStatus(prev => ({
      ...prev,
      is_processing: true,
      processing_message: "Uploading and transcribing media file..."
    }));

    const formData = new FormData();
    formData.append('file', mediaFile);
    formData.append('session_id', sessionId);

    try {
      const res = await fetch(`${API_BASE}/upload/media`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Media processing failed");

      setMediaFile(null);
      if (mediaInputRef.current) mediaInputRef.current.value = "";
      alert(data.message || "Media file indexed!");
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      fetchStatus();
    }
  };

  // --- Session Control Actions ---

  const handleClearHistory = async () => {
    if (!confirm("Are you sure you want to clear the chat history?")) return;
    try {
      const res = await fetch(`${API_BASE}/clear-history`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      });
      if (res.ok) {
        setChatHistory([]);
        alert("Chat history cleared!");
      }
    } catch (e) {
      alert("Failed to clear chat history");
    }
  };

  const handleResetWorkspace = async () => {
    if (!confirm("Reset workspace? This removes all active PDFs, URLs, audio, and chat logs.")) return;
    try {
      const res = await fetch(`${API_BASE}/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      });
      if (res.ok) {
        setChatHistory([]);
        setWorkspaceStatus({
          document_ids: [],
          has_active_image: false,
          active_image_name: "",
          message_count: 0,
          is_processing: false,
          processing_message: "",
          has_summary: false,
          has_quiz: false,
          has_comparison: false
        });
        setActiveAnalysisMode(null);
        setSummaryData("");
        setComparisonData("");
        setQuizData(null);
        setQuizAnswers({});
        setQuizRevealed({});
        alert("Workspace reset successful!");
      }
    } catch (e) {
      alert("Failed to reset workspace");
    }
  };

  const handleRemoveSource = async (docId) => {
    if (!confirm(`Are you sure you want to remove "${docId}" from your sources?`)) return;
    
    try {
      const res = await fetch(`${API_BASE}/remove-source`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, document_id: docId })
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to remove source");
      
      alert(data.message || "Source successfully removed!");
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      fetchStatus();
    }
  };

  // --- Sidebar/Action Buttons click events ---

  const handleGenerateSummary = async () => {
    setActiveAnalysisMode('summary');
    setMobileActivePanel('guide');
    setSummaryData("### Summarizing Knowledge base...\nGenerating themes. This may take a few seconds.");
    try {
      const res = await fetch(`${API_BASE}/action/summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to generate summary");
      setSummaryData(data.summary);
    } catch (err) {
      setSummaryData(`### ⚠️ Failed to generate summary\n${err.message}`);
    }
  };

  const handleGenerateQuiz = async () => {
    setActiveAnalysisMode('quiz');
    setMobileActivePanel('guide');
    setQuizData(null);
    setQuizAnswers({});
    setQuizRevealed({});
    try {
      const res = await fetch(`${API_BASE}/action/quiz`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to generate quiz");
      setQuizData(data.quiz);
    } catch (err) {
      alert(`Failed to generate quiz: ${err.message}`);
      setActiveAnalysisMode(null);
    }
  };

  const handleCompareAssets = async () => {
    setActiveAnalysisMode('compare');
    setMobileActivePanel('guide');
    setComparisonData("### Building comparative matrix...\nSynthesizing loaded documents into grid fields.");
    try {
      const res = await fetch(`${API_BASE}/action/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to generate comparative matrix");
      setComparisonData(data.comparison);
    } catch (err) {
      setComparisonData(`### ⚠️ Failed to compare assets\n${err.message}`);
    }
  };

  // --- Chat Actions ---

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!userInput.trim() || workspaceStatus.is_processing || isTyping) return;

    const queryText = userInput;
    setUserInput('');
    setIsTyping(true);

    // Pessimist append user message
    setChatHistory(prev => [...prev, { role: "user", content: queryText, references: [] }]);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText, session_id: sessionId })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to get chatbot response");

      setChatHistory(prev => [...prev, { role: "assistant", content: data.answer, references: data.references || [] }]);
    } catch (err) {
      setChatHistory(prev => [...prev, { role: "assistant", content: `⚠️ Error processing request:\n${err.message}`, references: [] }]);
    } finally {
      setIsTyping(false);
      fetchStatus();
    }
  };

  // Quiz interactive marking helpers
  const handleSelectQuizOption = (qIdx, opt) => {
    setQuizAnswers(prev => ({ ...prev, [qIdx]: opt }));
  };

  const handleRevealQuizAnswer = (qIdx) => {
    setQuizRevealed(prev => ({ ...prev, [qIdx]: true }));
  };

  return (
    <div className="app-container">
      {/* Mobile background overlay for sidebar */}
      <div 
        className={`sidebar-overlay ${sidebarOpen ? 'visible' : ''}`} 
        onClick={() => setSidebarOpen(false)} 
      />

      {/* ================= SIDEBAR PANEL ================= */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-brand">
          <h2 className="sidebar-title">
            <span>🧠</span> Curio AI
          </h2>
        </div>

        <div className="sidebar-section-title">
          <span>⚙️</span> Ingest Workspace
        </div>

        {/* Collapsible Accordion for Adding Sources */}
        <div className="add-source-accordion">
          <button 
            className="add-source-trigger"
            onClick={() => setIngestDrawerOpen(prev => !prev)}
            title="Expand/Collapse Source Uploader"
          >
            <div className="add-source-trigger-content">
              <Plus size={14} />
              <span>Add Source Material</span>
            </div>
            {ingestDrawerOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          
          {ingestDrawerOpen && (
            <div className="add-source-content">
              {/* Tab Selectors */}
              <div className="sidebar-tabs">
                <button 
                  className={`sidebar-tab-btn ${activeTab === 'url' ? 'active' : ''}`}
                  onClick={() => setActiveTab('url')}
                >
                  <Youtube size={14} />
                  <span>YouTube</span>
                </button>
                <button 
                  className={`sidebar-tab-btn ${activeTab === 'pdf' ? 'active' : ''}`}
                  onClick={() => setActiveTab('pdf')}
                >
                  <FileText size={14} />
                  <span>PDFs</span>
                </button>
                <button 
                  className={`sidebar-tab-btn ${activeTab === 'image' ? 'active' : ''}`}
                  onClick={() => setActiveTab('image')}
                >
                  <ImageIcon size={14} />
                  <span>Image</span>
                </button>
                <button 
                  className={`sidebar-tab-btn ${activeTab === 'media' ? 'active' : ''}`}
                  onClick={() => setActiveTab('media')}
                >
                  <Music size={14} />
                  <span>Audio/Video</span>
                </button>
              </div>

              {/* Tab Content Panels */}
              <div className="tab-content">
                {activeTab === 'url' && (
                  <form onSubmit={handleYtSubmit} className="input-group">
                    <span className="input-label">Paste YouTube URLs</span>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                      {ytUrls.map((url, index) => (
                        <div key={index} style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                          <input 
                            type="text"
                            className="text-area"
                            style={{ flex: 1, padding: '0.4rem 0.5rem', minHeight: '34px', fontSize: '0.78rem' }}
                            placeholder="https://youtube.com/watch?v=..."
                            value={url}
                            onChange={(e) => {
                              const newUrls = [...ytUrls];
                              newUrls[index] = e.target.value;
                              setYtUrls(newUrls);
                            }}
                          />
                          {ytUrls.length > 1 && (
                            <button 
                              type="button"
                              className="btn-outline"
                              style={{ padding: 0, height: '34px', width: '34px', minWidth: '34px', display: 'flex', alignItems: 'center', justifyContent: 'center', borderColor: 'var(--border-color)', color: 'var(--error)' }}
                              onClick={() => {
                                const newUrls = ytUrls.filter((_, idx) => idx !== index);
                                setYtUrls(newUrls);
                              }}
                              title="Remove this video link field"
                            >
                              <X size={14} />
                            </button>
                          )}
                        </div>
                      ))}
                      <button 
                        type="button"
                        className="btn-outline"
                        style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.3rem', fontSize: '0.74rem', padding: '6px 12px', borderStyle: 'dashed' }}
                        onClick={() => setYtUrls([...ytUrls, ''])}
                      >
                        <Plus size={12} />
                        <span>Add another video</span>
                      </button>
                    </div>
                    <button 
                      type="submit" 
                      className="btn-primary"
                      disabled={workspaceStatus.is_processing || !ytUrls.some(u => u.trim())}
                    >
                      <Youtube size={14} /> Link Video
                    </button>
                  </form>
                )}

                {activeTab === 'pdf' && (
                  <div className="input-group">
                    <span className="input-label">Upload PDFs</span>
                    <div 
                      className="dropzone"
                      onClick={() => pdfInputRef.current?.click()}
                    >
                      <FileText size={24} className="active-resource-icon" />
                      <p>Click to select PDFs</p>
                      <span>Supports multiple files</span>
                    </div>
                    <input 
                      type="file"
                      ref={pdfInputRef}
                      style={{ display: 'none' }}
                      accept=".pdf"
                      multiple
                      onChange={(e) => setPdfFiles(Array.from(e.target.files))}
                    />
                    {pdfFiles.length > 0 && (
                      <div>
                        <div className="file-name-preview">
                          <FileCheck size={12} /> Selected: {pdfFiles.length} file(s)
                        </div>
                        <button 
                          className="btn-primary"
                          onClick={handlePdfSubmit}
                          disabled={workspaceStatus.is_processing}
                        >
                          🚀 Process Documents
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {activeTab === 'image' && (
                  <div className="input-group">
                    <span className="input-label">Mount Image Resource</span>
                    <div 
                      className="dropzone"
                      onClick={() => imgInputRef.current?.click()}
                    >
                      <ImageIcon size={24} className="active-resource-icon" />
                      <p>Select image file</p>
                      <span>PNG, JPG, JPEG</span>
                    </div>
                    <input 
                      type="file"
                      ref={imgInputRef}
                      style={{ display: 'none' }}
                      accept="image/*"
                      onChange={(e) => setImageFile(e.target.files[0])}
                    />
                    {imageFile && (
                      <div>
                        <div className="file-name-preview">
                          <FileCheck size={12} /> {imageFile.name}
                        </div>
                        <button 
                          className="btn-primary"
                          onClick={handleImageSubmit}
                          disabled={workspaceStatus.is_processing}
                        >
                          👁️ Ingest Image
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {activeTab === 'media' && (
                  <div className="input-group">
                    <span className="input-label">Transcribe Local Media</span>
                    <div 
                      className="dropzone"
                      onClick={() => mediaInputRef.current?.click()}
                    >
                      <Music size={24} className="active-resource-icon" />
                      <p>Choose Audio/Video</p>
                      <span>mp3, wav, m4a, mp4</span>
                    </div>
                    <input 
                      type="file"
                      ref={mediaInputRef}
                      style={{ display: 'none' }}
                      accept=".mp3,.wav,.m4a,.mp4"
                      onChange={(e) => setMediaFile(e.target.files[0])}
                    />
                    {mediaFile && (
                      <div>
                        <div className="file-name-preview">
                          <FileCheck size={12} /> {mediaFile.name}
                        </div>
                        <button 
                          className="btn-primary"
                          onClick={handleMediaSubmit}
                          disabled={workspaceStatus.is_processing}
                        >
                          ⚡ Transcribe Asset
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Processing Indicator */}
        {workspaceStatus.is_processing && (
          <div className="loader-container">
            <div className="pulse-spinner"></div>
            <p className="loader-text">{workspaceStatus.processing_message}</p>
          </div>
        )}

        {/* Active Resources Grid Tiles (NotebookLM style) */}
        <div className="sidebar-section-title">
          <span>📚</span> Active Sources
        </div>

        <div style={{ flex: 1, minHeight: '120px', overflowY: 'auto', marginBottom: '1rem' }}>
          {workspaceStatus.document_ids.length > 0 ? (
            <div className="active-resource-list">
              {workspaceStatus.document_ids.map((id, index) => {
                const isPdf = id.toLowerCase().endsWith('.pdf');
                const isAudio = id.toLowerCase().endsWith('.mp3') || id.toLowerCase().endsWith('.wav') || id.toLowerCase().endsWith('.m4a') || id.toLowerCase().endsWith('.mp4');
                const isImg = workspaceStatus.has_active_image && workspaceStatus.active_image_name === id;

                return (
                  <div key={index} className="active-resource-card active" title={id}>
                    <div className="active-resource-top">
                      <div className="active-resource-icon">
                        {isImg ? (
                          <ImageIcon size={12} />
                        ) : isPdf ? (
                          <FileText size={12} />
                        ) : isAudio ? (
                          <Music size={12} />
                        ) : (
                          <Youtube size={12} />
                        )}
                      </div>
                      <div className="active-resource-indicator" title="Source loaded and active" />
                    </div>
                    <span className="active-resource-name">{id}</span>
                    <button 
                      className="active-resource-close-btn"
                      title="Remove Source"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRemoveSource(id);
                      }}
                    >
                      <X size={10} />
                    </button>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="empty-resource-info">
              No active learning resources mounted. Load material to unlock learning accelerators.
            </div>
          )}
        </div>

        {/* Reset / History Controls */}
        <div className="sidebar-footer">
          <button 
            className="btn-outline btn-danger"
            onClick={handleClearHistory}
            disabled={workspaceStatus.is_processing || chatHistory.length === 0}
          >
            <Trash2 size={12} /> Clear Chat
          </button>
          <button 
            className="btn-outline"
            style={{ borderColor: 'rgba(239, 68, 68, 0.3)', color: 'var(--error)' }}
            onClick={handleResetWorkspace}
            disabled={workspaceStatus.is_processing}
          >
            <RotateCcw size={12} /> Reset Workspace
          </button>
        </div>
      </aside>

      {/* ================= MAIN VIEWPORT ================= */}
      <main className="main-viewport">
        {/* Header Section */}
        <header className="main-header">
          <div className="header-meta">
            <button 
              className="hamburger-btn"
              onClick={() => setSidebarOpen(prev => !prev)}
              title="Toggle Sidebar"
            >
              <Menu size={20} />
            </button>
            <div className="header-title-wrapper">
              <h1 className="header-title">Curio AI</h1>
              <p className="header-subtitle">
                Advanced RAG context pipeline powered by Pinecone & Llama 3
              </p>
            </div>
          </div>

          <div className="header-actions">
            {/* Core Accelerator Actions */}
            <div className="action-buttons-container">
              <button 
                className="action-btn"
                disabled={workspaceStatus.document_ids.length === 0 || workspaceStatus.is_processing}
                onClick={handleGenerateSummary}
                title="Generate Workspace Study Summary"
              >
                <Sparkles size={13} style={{ color: 'var(--accent)' }} />
                <span>AI Summary</span>
              </button>
              <button 
                className="action-btn"
                disabled={workspaceStatus.document_ids.length === 0 || workspaceStatus.is_processing}
                onClick={handleGenerateQuiz}
                title="Generate Analytical Evaluation Quiz"
              >
                <BookOpen size={13} style={{ color: 'var(--success)' }} />
                <span>Generate Quiz</span>
              </button>
              <button 
                className="action-btn"
                disabled={workspaceStatus.document_ids.length < 2 || workspaceStatus.is_processing || workspaceStatus.has_active_image}
                onClick={handleCompareAssets}
                title="Compare active workspace documents"
              >
                <Layers size={13} style={{ color: 'var(--secondary)' }} />
                <span>Compare Assets</span>
              </button>
            </div>

            {/* Dark/Light Mode Theme Switcher */}
            <button 
              className="theme-toggle-btn"
              onClick={toggleTheme}
              title={theme === 'dark' ? "Switch to Light Mode" : "Switch to Dark Mode"}
            >
              {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
            </button>
          </div>
        </header>

        {/* Mobile Split Layout Tab Switcher */}
        {activeAnalysisMode && (
          <div className="mobile-split-tabs">
            <button 
              className={`mobile-split-tab-btn ${mobileActivePanel === 'chat' ? 'active' : ''}`}
              onClick={() => setMobileActivePanel('chat')}
            >
              Chat Session
            </button>
            <button 
              className={`mobile-split-tab-btn ${mobileActivePanel === 'guide' ? 'active' : ''}`}
              onClick={() => setMobileActivePanel('guide')}
            >
              {activeAnalysisMode === 'summary' ? 'Study Summary' : activeAnalysisMode === 'quiz' ? 'Evaluation Quiz' : 'Comparison Matrix'}
            </button>
          </div>
        )}

        {/* Center Workspace Area */}
        <div className="workspace-split">
          
          {/* Chat Panel */}
          <section className={`chat-section ${(!activeAnalysisMode || mobileActivePanel === 'chat') ? 'active' : ''}`}>
            <div className="chat-messages-container">
              {chatHistory.length === 0 ? (
                <div className="chat-welcome-box">
                  <div className="welcome-icon-glow">
                    <Sparkles size={24} />
                  </div>
                  <h3>Ready to Learn?</h3>
                  <p>
                    Load lecture PDFs, paste YouTube lecture URLs, upload audio recordings, or input screenshots in the sidebar workspace.<br/>
                    I will index their semantics so we can run hybrid keyword/vector search and discuss concepts with citations.
                  </p>
                </div>
              ) : (
                chatHistory.map((msg, index) => (
                  <div key={index} className={`message-bubble ${msg.role}`}>
                    <div className="message-avatar">
                      {msg.role === 'user' ? '👨‍🎓' : '🤖'}
                    </div>
                    <div className="message-content-wrapper">
                      <div 
                        className="message-content"
                        dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.content) }}
                      />
                      {/* Source References */}
                      {msg.references && msg.references.length > 0 && (
                        <div className="message-references">
                          {msg.references.map((ref, rIdx) => {
                            // Detect if the ref is a clickable URL
                            const urlMatch = ref.match(/\[(.*?)\]\((.*?)\)/);
                            if (urlMatch) {
                              const label = urlMatch[1];
                              const href = urlMatch[2];
                              return (
                                <a 
                                  key={rIdx} 
                                  href={href} 
                                  target="_blank" 
                                  rel="noopener noreferrer" 
                                  className="ref-tag"
                                >
                                  <ExternalLink size={10} /> {label}
                                </a>
                              );
                            }
                            return (
                              <span key={rIdx} className="ref-tag">
                                <FileCheck size={10} /> {ref}
                              </span>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
              {isTyping && (
                <div className="message-bubble assistant">
                  <div className="message-avatar">🤖</div>
                  <div className="message-content-wrapper">
                    <div className="message-content" style={{ padding: '0.6rem' }}>
                      <div className="typing-indicator">
                        <div className="typing-dot"></div>
                        <div className="typing-dot"></div>
                        <div className="typing-dot"></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Chat inputs */}
            <form onSubmit={handleSendMessage} className="chat-input-container">
              <input 
                className="chat-textarea"
                placeholder={workspaceStatus.document_ids.length > 0 ? "Ask anything about loaded materials..." : "🚨 Load workspace resources before chatting"}
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                disabled={workspaceStatus.document_ids.length === 0 || workspaceStatus.is_processing || isTyping}
              />
              <button 
                type="submit" 
                className="chat-send-btn"
                disabled={!userInput.trim() || workspaceStatus.is_processing || isTyping}
              >
                <Send size={14} />
              </button>
            </form>
          </section>

          {/* Analysis Right Accelerator Panel (only if active) */}
          {activeAnalysisMode && (
            <section className={`analysis-section ${(activeAnalysisMode && mobileActivePanel === 'guide') ? 'active' : ''}`}>
              <div className="analysis-header">
                <div className="analysis-title">
                  {activeAnalysisMode === 'summary' && (
                    <>
                      <Sparkles size={16} />
                      <span>Workspace Study Summary</span>
                    </>
                  )}
                  {activeAnalysisMode === 'quiz' && (
                    <>
                      <BookOpen size={16} />
                      <span>Analytical Knowledge Evaluation</span>
                    </>
                  )}
                  {activeAnalysisMode === 'compare' && (
                    <>
                      <Layers size={16} />
                      <span>Asset Comparison Matrix</span>
                    </>
                  )}
                </div>
                <button 
                  className="analysis-close-btn"
                  onClick={() => setActiveAnalysisMode(null)}
                >
                  <X size={16} />
                </button>
              </div>

              <div className="analysis-body">
                {activeAnalysisMode === 'summary' && (
                  <div dangerouslySetInnerHTML={{ __html: parseMarkdown(summaryData) }} />
                )}

                {activeAnalysisMode === 'compare' && (
                  <div dangerouslySetInnerHTML={{ __html: parseMarkdown(comparisonData) }} />
                )}

                {activeAnalysisMode === 'quiz' && (
                  <div className="quiz-container">
                    {!quizData ? (
                      <div className="chat-welcome-box" style={{ margin: '3rem auto' }}>
                        <div className="pulse-spinner"></div>
                        <p className="loader-text">Compiling quiz questions from documents...</p>
                      </div>
                    ) : (
                      quizData.quiz.map((q, qIdx) => {
                        const selected = quizAnswers[qIdx];
                        const revealed = quizRevealed[qIdx];
                        return (
                          <div key={qIdx} className="quiz-card">
                            <p className="quiz-question">Q{qIdx + 1}: {q.question}</p>
                            <div className="quiz-options">
                              {q.options.map((opt, oIdx) => {
                                const isCorrect = opt === q.correct_answer;
                                const isSelected = selected === opt;
                                
                                let optionClass = "";
                                if (revealed) {
                                  if (isCorrect) optionClass = "correct";
                                  else if (isSelected) optionClass = "incorrect";
                                }

                                return (
                                  <label key={oIdx} className={`quiz-option-label ${optionClass}`}>
                                    <input 
                                      type="radio" 
                                      name={`question_${qIdx}`}
                                      className="quiz-option-input"
                                      disabled={revealed}
                                      checked={isSelected}
                                      onChange={() => handleSelectQuizOption(qIdx, opt)}
                                    />
                                    <span>{opt}</span>
                                  </label>
                                );
                              })}
                            </div>
                            
                            {!revealed ? (
                              <button 
                                className="btn-primary" 
                                style={{ margin: '0.4rem 0 0 0', padding: '6px 12px', fontSize: '0.8rem', width: 'auto', alignSelf: 'flex-start' }}
                                disabled={!selected}
                                onClick={() => handleRevealQuizAnswer(qIdx)}
                              >
                                Check Answer
                              </button>
                            ) : (
                              <div className="quiz-explanation-box">
                                <p style={{ fontWeight: 600, color: selected === q.correct_answer ? 'var(--success)' : 'var(--error)', marginBottom: '0.2rem', fontSize: '0.78rem' }}>
                                  {selected === q.correct_answer ? "✓ Correct Option!" : `✗ Incorrect. Correct choice: ${q.correct_answer}`}
                                </p>
                                <p>{q.explanation}</p>
                              </div>
                            )}
                          </div>
                        );
                      })
                    )}
                  </div>
                )}
              </div>
            </section>
          )}

        </div>
      </main>
    </div>
  );
}
