/**
 * Flux Neural Gesture Classifier
 * Deep Learning MLP that classifies hand landmark configurations into named gestures.
 * Uses TensorFlow.js for on-device inference — no server needed.
 * 
 * Architecture: 63 inputs (21 landmarks × 3 axes) → 128 → 64 → 8 classes (softmax)
 * Training: Runs a micro-training cycle on synthetic landmark data at initialization.
 */
class GestureClassifier {
  constructor() {
    this.model = null;
    this.loaded = false;
    this.labels = [
      'open_palm',     // All fingers extended
      'fist',          // All fingers curled
      'point',         // Only index extended
      'peace',         // Index + middle extended (V sign)
      'thumbs_up',     // Only thumb extended
      'rock',          // Index + pinky extended 🤘
      'pinch',         // Thumb + index touching
      'three'          // Index + middle + ring extended
    ];

    // Display names with emoji
    this.displayNames = {
      'open_palm': '🖐️ Open Palm',
      'fist': '✊ Fist',
      'point': '☝️ Point',
      'peace': '✌️ Peace',
      'thumbs_up': '👍 Thumbs Up',
      'rock': '🤘 Rock',
      'pinch': '🤏 Pinch',
      'three': '🤟 Three'
    };

    // Gesture → action mapping
    this.gestureActions = {
      'peace': 'screenshot',
      'fist': 'clear_canvas',
      'thumbs_up': 'save_session',
      'rock': 'toggle_audio',
      'three': 'cycle_color'
    };

    // Smoothing buffer for temporal stability (prevents flickering)
    this.predictionBuffer = [];
    this.bufferSize = 5;
    this.lastGesture = null;
    this.lastConfidence = 0;
    this.gestureHoldFrames = 0;
    this.actionCooldown = {};
  }

  /**
   * Initialize the classifier with TensorFlow.js
   * Builds and trains a lightweight MLP on synthetic gesture data
   */
  async init() {
    if (this.loaded) return;

    try {
      // Wait for TF.js to be available
      if (typeof tf === 'undefined') {
        console.warn('GestureClassifier: TF.js not loaded, using heuristic fallback');
        this.loaded = true;
        this.useFallback = true;
        return;
      }

      console.log('GestureClassifier: Building neural gesture model...');

      // Build a small MLP
      this.model = tf.sequential();
      this.model.add(tf.layers.dense({
        inputShape: [63], // 21 landmarks × 3 (x, y, z)
        units: 128,
        activation: 'relu',
        kernelInitializer: 'glorotUniform'
      }));
      this.model.add(tf.layers.dropout({ rate: 0.2 }));
      this.model.add(tf.layers.dense({
        units: 64,
        activation: 'relu'
      }));
      this.model.add(tf.layers.dropout({ rate: 0.15 }));
      this.model.add(tf.layers.dense({
        units: this.labels.length,
        activation: 'softmax'
      }));

      this.model.compile({
        optimizer: tf.train.adam(0.002),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });

      // Generate synthetic training data and train
      const { xs, ys } = this._generateTrainingData();
      
      await this.model.fit(xs, ys, {
        epochs: 40,
        batchSize: 32,
        shuffle: true,
        verbose: 0
      });

      // Cleanup training tensors
      xs.dispose();
      ys.dispose();

      this.loaded = true;
      this.useFallback = false;
      console.log('GestureClassifier: Neural model trained and ready.');
    } catch (e) {
      console.error('GestureClassifier: Init failed, using heuristic fallback', e);
      this.loaded = true;
      this.useFallback = true;
    }
  }

  /**
   * Generate synthetic landmark data for training
   * Creates realistic hand poses for each gesture class
   */
  _generateTrainingData() {
    const samplesPerClass = 120;
    const data = [];
    const labels = [];

    // MediaPipe landmark layout reference:
    // 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
    // Each landmark: {x: 0-1, y: 0-1, z: -0.3 to 0.3}

    for (let s = 0; s < samplesPerClass; s++) {
      const noise = () => (Math.random() - 0.5) * 0.04;
      const jitter = () => (Math.random() - 0.5) * 0.02;

      // ── OPEN PALM: all fingers extended upward ──
      const openPalm = this._makeHandPose({
        thumb: 'extended', index: 'extended', middle: 'extended',
        ring: 'extended', pinky: 'extended'
      }, noise);
      data.push(openPalm);
      labels.push(0);

      // ── FIST: all fingers curled ──
      const fist = this._makeHandPose({
        thumb: 'curled', index: 'curled', middle: 'curled',
        ring: 'curled', pinky: 'curled'
      }, noise);
      data.push(fist);
      labels.push(1);

      // ── POINT: index extended, rest curled ──
      const point = this._makeHandPose({
        thumb: 'curled', index: 'extended', middle: 'curled',
        ring: 'curled', pinky: 'curled'
      }, noise);
      data.push(point);
      labels.push(2);

      // ── PEACE: index + middle extended ──
      const peace = this._makeHandPose({
        thumb: 'curled', index: 'extended', middle: 'extended',
        ring: 'curled', pinky: 'curled'
      }, noise);
      data.push(peace);
      labels.push(3);

      // ── THUMBS UP: only thumb extended ──
      const thumbsUp = this._makeHandPose({
        thumb: 'extended', index: 'curled', middle: 'curled',
        ring: 'curled', pinky: 'curled'
      }, noise);
      data.push(thumbsUp);
      labels.push(4);

      // ── ROCK: index + pinky extended 🤘 ──
      const rock = this._makeHandPose({
        thumb: 'curled', index: 'extended', middle: 'curled',
        ring: 'curled', pinky: 'extended'
      }, noise);
      data.push(rock);
      labels.push(5);

      // ── PINCH: thumb and index close together ──
      const pinch = this._makeHandPose({
        thumb: 'pinch', index: 'pinch', middle: 'curled',
        ring: 'curled', pinky: 'curled'
      }, noise);
      data.push(pinch);
      labels.push(6);

      // ── THREE: index + middle + ring extended ──
      const three = this._makeHandPose({
        thumb: 'curled', index: 'extended', middle: 'extended',
        ring: 'extended', pinky: 'curled'
      }, noise);
      data.push(three);
      labels.push(7);
    }

    const xs = tf.tensor2d(data);
    const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), this.labels.length);

    return { xs, ys };
  }

  /**
   * Generate a synthetic 21-landmark hand pose
   * @param {Object} config - finger states (extended/curled/pinch)
   * @param {Function} noise - noise function for data augmentation
   * @returns {number[]} - flat array of 63 values (21×3)
   */
  _makeHandPose(config, noise) {
    const landmarks = [];

    // Wrist (anchor point)
    const wx = 0.5 + noise(), wy = 0.75 + noise(), wz = 0 + noise();
    landmarks.push(wx, wy, wz);

    // Helper to generate finger landmarks
    const makeFinger = (baseX, baseY, state, tipSpread = 0) => {
      const points = [];
      if (state === 'extended') {
        // MCP (base)
        points.push(baseX + noise(), baseY - 0.05 + noise(), noise() * 0.5);
        // PIP
        points.push(baseX + noise(), baseY - 0.12 + noise(), noise() * 0.5);
        // DIP
        points.push(baseX + tipSpread + noise(), baseY - 0.18 + noise(), noise() * 0.5);
        // TIP (far from wrist)
        points.push(baseX + tipSpread + noise(), baseY - 0.25 + noise(), noise() * 0.5);
      } else if (state === 'curled') {
        // MCP
        points.push(baseX + noise(), baseY - 0.04 + noise(), noise() * 0.5);
        // PIP (folded back toward palm)
        points.push(baseX + noise(), baseY - 0.02 + noise(), 0.05 + noise() * 0.5);
        // DIP
        points.push(baseX + noise(), baseY + 0.02 + noise(), 0.08 + noise() * 0.5);
        // TIP (close to palm)
        points.push(baseX + noise(), baseY + 0.04 + noise(), 0.1 + noise() * 0.5);
      } else if (state === 'pinch') {
        // MCP
        points.push(baseX + noise(), baseY - 0.04 + noise(), noise() * 0.5);
        // PIP
        points.push(baseX + noise(), baseY - 0.08 + noise(), noise() * 0.5);
        // DIP
        points.push(0.48 + noise(), baseY - 0.10 + noise(), noise() * 0.5);
        // TIP (converging to pinch point)
        points.push(0.47 + noise(), baseY - 0.12 + noise(), noise() * 0.5);
      }
      return points;
    };

    // Thumb (landmarks 1-4)
    landmarks.push(...makeFinger(wx - 0.08, wy, config.thumb, -0.03));
    // Index (landmarks 5-8)
    landmarks.push(...makeFinger(wx - 0.04, wy, config.index, -0.01));
    // Middle (landmarks 9-12)
    landmarks.push(...makeFinger(wx, wy, config.middle, 0));
    // Ring (landmarks 13-16)
    landmarks.push(...makeFinger(wx + 0.04, wy, config.ring, 0.01));
    // Pinky (landmarks 17-20)
    landmarks.push(...makeFinger(wx + 0.08, wy, config.pinky, 0.02));

    return landmarks;
  }

  /**
   * Classify a hand's landmark array into a named gesture
   * @param {Array} hand - MediaPipe hand landmarks (21 points, each {x, y, z})
   * @returns {Object} - { gesture, confidence, displayName, action }
   */
  async classify(hand) {
    if (!hand || !this.loaded) return null;

    if (this.useFallback) {
      return this._heuristicClassify(hand);
    }

    try {
      // Flatten landmarks to [x0,y0,z0, x1,y1,z1, ...]
      const flat = [];
      for (const lm of hand) {
        flat.push(lm.x, lm.y, lm.z);
      }

      const inputTensor = tf.tensor2d([flat]);
      const prediction = this.model.predict(inputTensor);
      const probs = await prediction.data();

      inputTensor.dispose();
      prediction.dispose();

      // Find top prediction
      let maxIdx = 0, maxProb = 0;
      for (let i = 0; i < probs.length; i++) {
        if (probs[i] > maxProb) {
          maxProb = probs[i];
          maxIdx = i;
        }
      }

      const gesture = this.labels[maxIdx];
      const confidence = Math.round(maxProb * 100);

      // Temporal smoothing
      return this._smoothPrediction(gesture, confidence);
    } catch (e) {
      return this._heuristicClassify(hand);
    }
  }

  /**
   * Heuristic fallback when TF.js is unavailable
   */
  _heuristicClassify(hand) {
    const fingerStates = [];
    const FINGERTIP_IDS = [4, 8, 12, 16, 20];
    const FINGER_PIP_IDS = [3, 6, 10, 14, 18];

    for (let i = 0; i < 5; i++) {
      const tip = hand[FINGERTIP_IDS[i]];
      const pip = hand[FINGER_PIP_IDS[i]];
      const wrist = hand[0];

      if (i === 0) {
        // Thumb: check angle
        const a = hand[2], b = hand[3], c = hand[4];
        const bax = a.x - b.x, bay = a.y - b.y;
        const bcx = c.x - b.x, bcy = c.y - b.y;
        const dot = bax * bcx + bay * bcy;
        const mag = Math.sqrt(bax*bax + bay*bay) * Math.sqrt(bcx*bcx + bcy*bcy);
        const angle = mag > 0 ? Math.acos(Math.min(1, Math.max(-1, dot / mag))) : 0;
        fingerStates.push(angle > 2.2);
      } else {
        const tipDist = Math.sqrt(Math.pow(tip.x - wrist.x, 2) + Math.pow(tip.y - wrist.y, 2));
        const pipDist = Math.sqrt(Math.pow(pip.x - wrist.x, 2) + Math.pow(pip.y - wrist.y, 2));
        fingerStates.push(tipDist > pipDist * 1.05);
      }
    }

    const [thumb, index, middle, ring, pinky] = fingerStates;

    // Check pinch (thumb+index distance)
    const pinchDist = Math.sqrt(
      Math.pow(hand[4].x - hand[8].x, 2) + 
      Math.pow(hand[4].y - hand[8].y, 2) +
      Math.pow(hand[4].z - hand[8].z, 2)
    );

    let gesture, confidence;

    if (pinchDist < 0.06 && !middle && !ring) {
      gesture = 'pinch'; confidence = 85;
    } else if (thumb && index && middle && ring && pinky) {
      gesture = 'open_palm'; confidence = 90;
    } else if (!thumb && !index && !middle && !ring && !pinky) {
      gesture = 'fist'; confidence = 88;
    } else if (!thumb && index && !middle && !ring && !pinky) {
      gesture = 'point'; confidence = 85;
    } else if (!thumb && index && middle && !ring && !pinky) {
      gesture = 'peace'; confidence = 87;
    } else if (thumb && !index && !middle && !ring && !pinky) {
      gesture = 'thumbs_up'; confidence = 82;
    } else if (!thumb && index && !middle && !ring && pinky) {
      gesture = 'rock'; confidence = 80;
    } else if (!thumb && index && middle && ring && !pinky) {
      gesture = 'three'; confidence = 83;
    } else {
      gesture = 'open_palm'; confidence = 40;
    }

    return this._smoothPrediction(gesture, confidence);
  }

  /**
   * Temporal smoothing: require consistent predictions over N frames
   */
  _smoothPrediction(gesture, confidence) {
    this.predictionBuffer.push({ gesture, confidence });
    if (this.predictionBuffer.length > this.bufferSize) {
      this.predictionBuffer.shift();
    }

    // Count votes
    const votes = {};
    let totalConf = 0;
    this.predictionBuffer.forEach(p => {
      votes[p.gesture] = (votes[p.gesture] || 0) + 1;
      if (p.gesture === gesture) totalConf += p.confidence;
    });

    // Find majority
    let topGesture = gesture, topVotes = 0;
    for (const [g, count] of Object.entries(votes)) {
      if (count > topVotes) {
        topVotes = count;
        topGesture = g;
      }
    }

    const avgConf = Math.round(totalConf / (votes[gesture] || 1));
    const stability = Math.round((topVotes / this.predictionBuffer.length) * 100);

    // Track hold duration
    if (topGesture === this.lastGesture) {
      this.gestureHoldFrames++;
    } else {
      this.gestureHoldFrames = 0;
    }

    this.lastGesture = topGesture;
    this.lastConfidence = avgConf;

    return {
      gesture: topGesture,
      confidence: avgConf,
      stability,
      displayName: this.displayNames[topGesture] || topGesture,
      action: this.gestureActions[topGesture] || null,
      holdFrames: this.gestureHoldFrames
    };
  }

  /**
   * Check if a gesture action should be triggered (with cooldown)
   * @param {string} gesture - the gesture name
   * @param {number} holdFrames - how many frames the gesture has been held
   * @returns {string|null} - action to execute, or null
   */
  checkAction(gesture, holdFrames) {
    const action = this.gestureActions[gesture];
    if (!action) return null;

    // Require 20 frames (~0.33s at 60fps) of hold before triggering
    if (holdFrames < 20) return null;

    // Cooldown: 2 seconds between same action
    const now = performance.now();
    if (this.actionCooldown[action] && (now - this.actionCooldown[action]) < 2000) {
      return null;
    }

    this.actionCooldown[action] = now;
    return action;
  }
}

// Global export
window.FluxGestureClassifier = new GestureClassifier();
