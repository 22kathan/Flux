/**
 * Flux Session Analytics Engine
 * Tracks gesture usage, mode switches, stroke data, and generates
 * LLM-style natural language session summaries using NLP templates.
 */
class SessionAnalytics {
  constructor() {
    this.sessionStart = null;
    this.sessionData = {
      gestures: {},        // gesture_name → count
      modeTime: {},        // mode_name → ms spent
      modeSwitches: 0,
      totalStrokes: 0,
      totalParticles: 0,
      voiceCommands: 0,
      screenshots: 0,
      peakFPS: 0,
      avgFPS: 0,
      fpsReadings: [],
      gestureTimeline: [],  // [{time, gesture, confidence}]
      commandLog: []        // [{time, command, source}]
    };
    this.currentMode = 'laser';
    this.modeStartTime = null;
    this.isActive = false;
  }

  /**
   * Start a new session
   */
  startSession() {
    this.sessionStart = Date.now();
    this.modeStartTime = Date.now();
    this.isActive = true;
    this.sessionData = {
      gestures: {},
      modeTime: { laser: 0, write: 0, os: 0 },
      modeSwitches: 0,
      totalStrokes: 0,
      totalParticles: 0,
      voiceCommands: 0,
      screenshots: 0,
      peakFPS: 0,
      avgFPS: 0,
      fpsReadings: [],
      gestureTimeline: [],
      commandLog: []
    };
    console.log('SessionAnalytics: Session started');
  }

  /**
   * Record a gesture detection
   */
  recordGesture(gesture, confidence) {
    if (!this.isActive) return;
    this.sessionData.gestures[gesture] = (this.sessionData.gestures[gesture] || 0) + 1;
    
    // Only record to timeline every 30th occurrence to avoid bloat
    const total = Object.values(this.sessionData.gestures).reduce((a, b) => a + b, 0);
    if (total % 30 === 0) {
      this.sessionData.gestureTimeline.push({
        time: Date.now() - this.sessionStart,
        gesture,
        confidence
      });
    }
  }

  /**
   * Record a mode switch
   */
  recordModeSwitch(newMode) {
    if (!this.isActive) return;
    
    // Record time in previous mode
    const now = Date.now();
    if (this.modeStartTime && this.currentMode) {
      const elapsed = now - this.modeStartTime;
      this.sessionData.modeTime[this.currentMode] = 
        (this.sessionData.modeTime[this.currentMode] || 0) + elapsed;
    }
    
    this.currentMode = newMode;
    this.modeStartTime = now;
    this.sessionData.modeSwitches++;
    
    this.sessionData.commandLog.push({
      time: now - this.sessionStart,
      command: `Switched to ${newMode} mode`,
      source: 'user'
    });
  }

  /**
   * Record FPS reading
   */
  recordFPS(fps) {
    if (!this.isActive) return;
    this.sessionData.fpsReadings.push(fps);
    if (fps > this.sessionData.peakFPS) {
      this.sessionData.peakFPS = fps;
    }
  }

  /**
   * Record a command (voice or gesture-triggered)
   */
  recordCommand(command, source = 'voice') {
    if (!this.isActive) return;
    if (source === 'voice') this.sessionData.voiceCommands++;
    
    this.sessionData.commandLog.push({
      time: Date.now() - this.sessionStart,
      command,
      source
    });
  }

  recordStroke() { if (this.isActive) this.sessionData.totalStrokes++; }
  recordScreenshot() { if (this.isActive) this.sessionData.screenshots++; }
  recordParticles(count) { if (this.isActive) this.sessionData.totalParticles += count; }

  /**
   * Generate a natural language session summary (LLM-style template generation)
   */
  generateSummary() {
    // Finalize current mode time
    if (this.modeStartTime && this.currentMode) {
      const elapsed = Date.now() - this.modeStartTime;
      this.sessionData.modeTime[this.currentMode] = 
        (this.sessionData.modeTime[this.currentMode] || 0) + elapsed;
    }

    const duration = Date.now() - this.sessionStart;
    const minutes = Math.floor(duration / 60000);
    const seconds = Math.floor((duration % 60000) / 1000);
    
    // Calculate average FPS
    const fpsArr = this.sessionData.fpsReadings;
    this.sessionData.avgFPS = fpsArr.length > 0
      ? Math.round(fpsArr.reduce((a, b) => a + b, 0) / fpsArr.length)
      : 0;

    // Determine dominant gesture
    const gestureCounts = Object.entries(this.sessionData.gestures);
    gestureCounts.sort((a, b) => b[1] - a[1]);
    const topGesture = gestureCounts.length > 0 ? gestureCounts[0] : ['none', 0];
    const totalGestures = gestureCounts.reduce((sum, [, count]) => sum + count, 0);

    // Determine dominant mode
    const modeTimes = Object.entries(this.sessionData.modeTime);
    modeTimes.sort((a, b) => b[1] - a[1]);
    const topMode = modeTimes.length > 0 ? modeTimes[0] : ['laser', 0];

    // ── NLP Template-based Summary Generation ──
    const modeNames = { laser: 'Laser Beams', write: 'Air Writing', os: 'OS Control' };
    const gestureDisplayNames = {
      'open_palm': 'open palm', 'fist': 'fist', 'point': 'pointing',
      'peace': 'peace sign', 'thumbs_up': 'thumbs up', 'rock': 'rock',
      'pinch': 'pinch', 'three': 'three fingers'
    };

    let summary = `## 📊 Flux Session Report\n\n`;
    summary += `**Duration:** ${minutes}m ${seconds}s\n\n`;
    
    // Performance paragraph
    summary += `### ⚡ Performance\n`;
    summary += `Your session maintained an average of **${this.sessionData.avgFPS} FPS** `;
    if (this.sessionData.avgFPS >= 80) {
      summary += `— excellent performance with buttery-smooth tracking. `;
    } else if (this.sessionData.avgFPS >= 50) {
      summary += `— good performance with stable hand tracking. `;
    } else {
      summary += `— below optimal. Consider closing other tabs for better performance. `;
    }
    summary += `Peak: **${this.sessionData.peakFPS} FPS**.\n\n`;

    // Mode usage
    summary += `### 🎮 Mode Usage\n`;
    summary += `You primarily used **${modeNames[topMode[0]] || topMode[0]}** mode, `;
    summary += `spending ${Math.round(topMode[1] / 1000)}s there. `;
    if (this.sessionData.modeSwitches > 3) {
      summary += `You switched modes **${this.sessionData.modeSwitches} times** — an active exploration of Flux's features! `;
    } else if (this.sessionData.modeSwitches > 0) {
      summary += `You switched modes ${this.sessionData.modeSwitches} time(s). `;
    }
    summary += '\n\n';

    // Gesture analysis
    if (totalGestures > 0) {
      summary += `### 🤚 Gesture Intelligence\n`;
      summary += `The neural gesture classifier detected **${totalGestures.toLocaleString()} gesture frames** across your session. `;
      summary += `Your most frequent gesture was **${gestureDisplayNames[topGesture[0]] || topGesture[0]}** `;
      summary += `(${topGesture[1].toLocaleString()} detections). `;
      
      if (gestureCounts.length > 2) {
        summary += `You also used ${gestureDisplayNames[gestureCounts[1][0]] || gestureCounts[1][0]} `;
        summary += `and ${gestureDisplayNames[gestureCounts[2]?.[0]] || 'other gestures'}. `;
      }
      summary += '\n\n';
    }

    // Creative output
    if (this.sessionData.totalStrokes > 0 || this.sessionData.screenshots > 0) {
      summary += `### 🎨 Creative Output\n`;
      if (this.sessionData.totalStrokes > 0) {
        summary += `You created **${this.sessionData.totalStrokes} stroke(s)** in air writing mode. `;
      }
      if (this.sessionData.screenshots > 0) {
        summary += `You captured **${this.sessionData.screenshots} screenshot(s)**. `;
      }
      summary += '\n\n';
    }

    // Voice commands
    if (this.sessionData.voiceCommands > 0) {
      summary += `### 🗣️ Voice Commands\n`;
      summary += `You issued **${this.sessionData.voiceCommands} voice command(s)** during this session. `;
      summary += `The NLP engine successfully interpreted your natural language instructions. `;
      summary += '\n\n';
    }

    // Particle effects
    if (this.sessionData.totalParticles > 100) {
      summary += `### ✨ Visual Effects\n`;
      summary += `A total of **${this.sessionData.totalParticles.toLocaleString()} particles** were spawned `;
      summary += `during your laser and drawing interactions. `;
      summary += '\n\n';
    }

    return {
      markdown: summary,
      data: {
        duration: { minutes, seconds, totalMs: duration },
        performance: { avgFPS: this.sessionData.avgFPS, peakFPS: this.sessionData.peakFPS },
        modes: this.sessionData.modeTime,
        modeSwitches: this.sessionData.modeSwitches,
        topGesture: { name: topGesture[0], count: topGesture[1] },
        totalGestures,
        strokes: this.sessionData.totalStrokes,
        screenshots: this.sessionData.screenshots,
        voiceCommands: this.sessionData.voiceCommands,
        particles: this.sessionData.totalParticles
      }
    };
  }
}

// Global export
window.FluxAnalytics = new SessionAnalytics();
