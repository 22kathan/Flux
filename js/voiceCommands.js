/**
 * Flux Voice Command Engine
 * NLP-powered voice control using Web Speech API + intent classification.
 * Supports natural language commands in English and Hindi.
 * 
 * Architecture: Speech → Tokenize → Intent Match (TF-IDF cosine + keyword boost) → Action
 */
class VoiceCommandEngine {
  constructor() {
    this.recognition = null;
    this.isListening = false;
    this.isSupported = false;
    this.onCommandCallback = null;
    this.onStatusCallback = null;
    this.onTranscriptCallback = null;
    this.language = 'en-US';
    this.confidenceThreshold = 0.35;

    // Intent database: each intent has example phrases and an action
    this.intents = [
      {
        name: 'switch_laser',
        action: 'mode_laser',
        phrases: [
          'switch to laser', 'laser mode', 'laser beams', 'turn on lasers',
          'show lasers', 'activate laser', 'beam mode', 'flux beams',
          'लेज़र मोड', 'लेज़र चालू करो'
        ],
        keywords: ['laser', 'beam', 'लेज़र']
      },
      {
        name: 'switch_write',
        action: 'mode_write',
        phrases: [
          'switch to writing', 'writing mode', 'draw mode', 'start drawing',
          'pen mode', 'activate drawing', 'write mode', 'air write',
          'लिखने का मोड', 'ड्राइंग मोड'
        ],
        keywords: ['write', 'writing', 'draw', 'drawing', 'pen', 'लिखन']
      },
      {
        name: 'switch_os',
        action: 'mode_os',
        phrases: [
          'switch to OS control', 'OS mode', 'mouse mode', 'cursor mode',
          'control mode', 'activate OS', 'desktop control',
          'ओएस मोड', 'माउस कंट्रोल'
        ],
        keywords: ['os', 'mouse', 'cursor', 'control', 'desktop', 'माउस']
      },
      {
        name: 'clear_canvas',
        action: 'clear',
        phrases: [
          'clear canvas', 'clear the screen', 'erase everything', 'clean up',
          'delete all', 'clear drawing', 'wipe canvas', 'reset canvas',
          'सब मिटाओ', 'कैनवास साफ करो'
        ],
        keywords: ['clear', 'erase', 'clean', 'wipe', 'delete', 'reset', 'मिटा', 'साफ']
      },
      {
        name: 'take_screenshot',
        action: 'screenshot',
        phrases: [
          'take screenshot', 'screenshot', 'capture screen', 'save image',
          'take a picture', 'snap', 'save screenshot',
          'स्क्रीनशॉट लो', 'फोटो लो'
        ],
        keywords: ['screenshot', 'capture', 'snap', 'picture', 'save', 'स्क्रीनशॉट', 'फोटो']
      },
      {
        name: 'fullscreen',
        action: 'fullscreen',
        phrases: [
          'go fullscreen', 'fullscreen', 'full screen', 'maximize',
          'enter fullscreen', 'exit fullscreen',
          'फुल स्क्रीन', 'पूरी स्क्रीन'
        ],
        keywords: ['fullscreen', 'full screen', 'maximize', 'फुल', 'पूर']
      },
      {
        name: 'change_color_cyan',
        action: 'color_0',
        phrases: [
          'change color to blue', 'blue color', 'cyan', 'light blue',
          'नीला रंग', 'नीला कलर'
        ],
        keywords: ['blue', 'cyan', 'नील']
      },
      {
        name: 'change_color_magenta',
        action: 'color_2',
        phrases: [
          'change color to pink', 'pink color', 'magenta', 'purple',
          'गुलाबी रंग', 'गुलाबी कलर'
        ],
        keywords: ['pink', 'magenta', 'purple', 'गुलाब']
      },
      {
        name: 'change_color_green',
        action: 'color_3',
        phrases: [
          'change color to green', 'green color', 'neon green',
          'हरा रंग', 'हरा कलर'
        ],
        keywords: ['green', 'हर']
      },
      {
        name: 'change_color_yellow',
        action: 'color_4',
        phrases: [
          'change color to yellow', 'yellow color',
          'पीला रंग', 'पीला कलर'
        ],
        keywords: ['yellow', 'पील']
      },
      {
        name: 'change_color_white',
        action: 'color_5',
        phrases: [
          'change color to white', 'white color',
          'सफेद रंग', 'सफेद कलर'
        ],
        keywords: ['white', 'सफेद']
      },
      {
        name: 'stop_listening',
        action: 'stop_voice',
        phrases: [
          'stop listening', 'stop voice', 'turn off voice', 'mute',
          'voice off', 'shut up', 'silence',
          'आवाज बंद करो', 'सुनना बंद करो'
        ],
        keywords: ['stop', 'mute', 'silence', 'off', 'बंद']
      }
    ];

    // Pre-compute TF-IDF vectors for all intent phrases
    this._precomputeIntentVectors();
  }

  /**
   * Initialize the Speech Recognition API
   */
  init() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      console.warn('VoiceCommandEngine: Speech Recognition not supported in this browser');
      this.isSupported = false;
      return false;
    }

    this.isSupported = true;
    this.recognition = new SpeechRecognition();
    this.recognition.continuous = true;
    this.recognition.interimResults = true;
    this.recognition.lang = this.language;
    this.recognition.maxAlternatives = 3;

    this.recognition.onresult = (event) => this._handleResult(event);
    this.recognition.onerror = (event) => this._handleError(event);
    this.recognition.onend = () => {
      // Auto-restart if we're supposed to be listening
      if (this.isListening) {
        try {
          this.recognition.start();
        } catch (e) {
          // Already started, ignore
        }
      }
    };

    console.log('VoiceCommandEngine: Initialized successfully');
    return true;
  }

  /**
   * Start listening for voice commands
   */
  start() {
    if (!this.isSupported || !this.recognition) {
      console.warn('VoiceCommandEngine: Cannot start — not supported');
      return;
    }

    if (this.isListening) return;

    try {
      this.isListening = true;
      this.recognition.start();
      if (this.onStatusCallback) {
        this.onStatusCallback({ listening: true, text: 'Listening...' });
      }
      console.log('VoiceCommandEngine: Started listening');
    } catch (e) {
      console.error('VoiceCommandEngine: Error starting', e);
    }
  }

  /**
   * Stop listening
   */
  stop() {
    if (!this.recognition) return;
    this.isListening = false;
    try {
      this.recognition.stop();
    } catch (e) {}
    if (this.onStatusCallback) {
      this.onStatusCallback({ listening: false, text: 'Voice off' });
    }
  }

  /**
   * Toggle listening state
   */
  toggle() {
    if (this.isListening) {
      this.stop();
    } else {
      this.start();
    }
  }

  /**
   * Set callbacks
   */
  onCommand(callback) { this.onCommandCallback = callback; }
  onStatus(callback) { this.onStatusCallback = callback; }
  onTranscript(callback) { this.onTranscriptCallback = callback; }

  // ══════════════════════════════════════
  //  NLP INTENT CLASSIFICATION
  // ══════════════════════════════════════

  /**
   * Pre-compute TF-IDF vectors for intent phrases
   */
  _precomputeIntentVectors() {
    this._stopwords = new Set([
      'a', 'an', 'the', 'is', 'are', 'to', 'of', 'in', 'on', 'at', 'for',
      'and', 'but', 'or', 'it', 'my', 'me', 'do', 'this', 'that'
    ]);

    // Build corpus
    this._allPhrases = [];
    this._phraseIntentMap = [];

    this.intents.forEach(intent => {
      intent.phrases.forEach(phrase => {
        this._allPhrases.push(phrase);
        this._phraseIntentMap.push(intent);
      });
    });

    // Build document frequency
    this._df = {};
    const tokenSets = this._allPhrases.map(p => {
      const tokens = this._tokenize(p);
      const unique = new Set(tokens);
      unique.forEach(t => { this._df[t] = (this._df[t] || 0) + 1; });
      return tokens;
    });

    // Build TF-IDF vectors
    const N = this._allPhrases.length;
    this._phraseVectors = tokenSets.map(tokens => {
      const tf = {};
      tokens.forEach(t => { tf[t] = (tf[t] || 0) + 1; });
      const len = tokens.length || 1;
      Object.keys(tf).forEach(t => { tf[t] /= len; });

      const tfidf = {};
      Object.keys(tf).forEach(t => {
        const idf = Math.log((N + 1) / (1 + (this._df[t] || 0)));
        tfidf[t] = tf[t] * idf;
      });
      return tfidf;
    });
  }

  _tokenize(text) {
    return text.toLowerCase()
      .replace(/[^a-z0-9\u0900-\u097F\u0A80-\u0AFF\s]/g, '')
      .split(/\s+/)
      .filter(w => w.length > 1 && !this._stopwords.has(w));
  }

  _cosineSimilarity(vecA, vecB) {
    const allKeys = new Set([...Object.keys(vecA), ...Object.keys(vecB)]);
    let dot = 0, magA = 0, magB = 0;
    allKeys.forEach(key => {
      const a = vecA[key] || 0;
      const b = vecB[key] || 0;
      dot += a * b;
      magA += a * a;
      magB += b * b;
    });
    const mag = Math.sqrt(magA) * Math.sqrt(magB);
    return mag === 0 ? 0 : dot / mag;
  }

  /**
   * Classify a transcript into an intent using TF-IDF cosine similarity + keyword boosting
   */
  classifyIntent(transcript) {
    const inputTokens = this._tokenize(transcript);
    if (inputTokens.length === 0) return null;

    // Build input TF-IDF vector
    const N = this._allPhrases.length;
    const tf = {};
    inputTokens.forEach(t => { tf[t] = (tf[t] || 0) + 1; });
    const len = inputTokens.length;
    Object.keys(tf).forEach(t => { tf[t] /= len; });

    const inputVec = {};
    Object.keys(tf).forEach(t => {
      const idf = Math.log((N + 1) / (1 + (this._df[t] || 0)));
      inputVec[t] = tf[t] * idf;
    });

    // Score each intent (aggregate best match from all its phrases)
    const intentScores = {};
    const lower = transcript.toLowerCase();

    this._phraseVectors.forEach((phraseVec, idx) => {
      const intent = this._phraseIntentMap[idx];
      const cosineSim = this._cosineSimilarity(inputVec, phraseVec);

      // Keyword boost
      let keywordBoost = 0;
      intent.keywords.forEach(kw => {
        if (lower.includes(kw.toLowerCase())) {
          keywordBoost += 0.25;
        }
      });

      const score = cosineSim * 0.6 + Math.min(keywordBoost, 0.5) * 0.4;

      if (!intentScores[intent.name] || score > intentScores[intent.name].score) {
        intentScores[intent.name] = {
          intent: intent,
          score,
          cosineSim,
          keywordBoost
        };
      }
    });

    // Find best intent
    let bestMatch = null;
    let bestScore = 0;

    Object.values(intentScores).forEach(m => {
      if (m.score > bestScore) {
        bestScore = m.score;
        bestMatch = m;
      }
    });

    if (bestMatch && bestScore >= this.confidenceThreshold) {
      return {
        intent: bestMatch.intent.name,
        action: bestMatch.intent.action,
        confidence: Math.round(bestScore * 100),
        transcript
      };
    }

    return null;
  }

  // ══════════════════════════════════════
  //  SPEECH EVENT HANDLERS
  // ══════════════════════════════════════

  _handleResult(event) {
    let finalTranscript = '';
    let interimTranscript = '';

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const result = event.results[i];
      if (result.isFinal) {
        finalTranscript += result[0].transcript;
      } else {
        interimTranscript += result[0].transcript;
      }
    }

    // Show interim results
    if (interimTranscript && this.onTranscriptCallback) {
      this.onTranscriptCallback({
        text: interimTranscript,
        isFinal: false
      });
    }

    // Process final results
    if (finalTranscript) {
      console.log('VoiceCommand: Heard:', finalTranscript);

      if (this.onTranscriptCallback) {
        this.onTranscriptCallback({
          text: finalTranscript,
          isFinal: true
        });
      }

      // Classify intent
      const match = this.classifyIntent(finalTranscript);
      if (match) {
        console.log(`VoiceCommand: Intent "${match.intent}" → Action "${match.action}" (${match.confidence}%)`);
        if (this.onCommandCallback) {
          this.onCommandCallback(match);
        }
      }
    }
  }

  _handleError(event) {
    if (event.error === 'no-speech') return; // Normal timeout
    if (event.error === 'aborted') return;   // Manual stop

    console.warn('VoiceCommand: Error:', event.error);
    if (this.onStatusCallback) {
      this.onStatusCallback({
        listening: this.isListening,
        text: `Error: ${event.error}`,
        error: true
      });
    }
  }
}

// Global export
window.FluxVoiceEngine = new VoiceCommandEngine();
