import * as THREE from 'three';

// ==========================================================================
// 3D Visualizer State
// ==========================================================================
let scene, camera, renderer, waveMesh, particleSystem;
let currentColor = new THREE.Color('#3b82f6');
let targetColor = new THREE.Color('#3b82f6');
let isDragging = false;
let previousMousePosition = { x: 0, y: 0 };

// ==========================================================================
// Audio Processing State
// ==========================================================================
let audioContext;
let analyserNode;
let audioSourceNode; // for file player
let micSourceNode;   // for live recording
let mediaRecorder;   // to record audio chunks
let recordedChunks = [];
let recordStartTime;
let recordTimerInterval;
let liveBlocks = []; // stores real-time analysis blocks
let animationFrameId;

// Analysis Results Cache
let currentFile = null;
let analyzedFiles = [];
let currentAnalysis = {
  score: 100,
  peakDbfs: -100,
  avgDbfs: -100,
  clippingBlocks: 0,
  deadAirDuration: 0,
  blocks: [],
  errors: []
};

// HTML Elements
const els = {
  webglContainer: document.getElementById('webgl-container'),
  statusIndicator: document.getElementById('engine-status'),
  tabUpload: document.getElementById('tab-upload'),
  tabRecord: document.getElementById('tab-record'),
  contentUpload: document.getElementById('content-upload'),
  contentRecord: document.getElementById('content-record'),
  dropZone: document.getElementById('drop-zone'),
  fileInput: document.getElementById('file-input'),
  browseBtn: document.getElementById('browse-btn'),
  fileInfoContainer: document.getElementById('file-info-container'),
  fileName: document.getElementById('file-name'),
  fileSize: document.getElementById('file-size'),
  analyzeBtn: document.getElementById('analyze-btn'),
  recordPulse: document.getElementById('record-pulse'),
  recordTimer: document.getElementById('record-timer'),
  recordStatusText: document.getElementById('record-status-text'),
  startRecordBtn: document.getElementById('start-record-btn'),
  stopRecordBtn: document.getElementById('stop-record-btn'),
  scorePanel: document.getElementById('score-panel'),
  scoreText: document.getElementById('score-text'),
  scoreRing: document.getElementById('score-progress-ring'),
  statPeakDb: document.getElementById('stat-peak-db'),
  statAvgDb: document.getElementById('stat-avg-db'),
  statClippingCount: document.getElementById('stat-clipping-count'),
  statDeadAirDuration: document.getElementById('stat-dead-air-duration'),
  verdictBanner: document.getElementById('verdict-banner'),
  verdictTitle: document.getElementById('verdict-title'),
  verdictDesc: document.getElementById('verdict-desc'),
  playerPanel: document.getElementById('player-panel'),
  audioPlayer: document.getElementById('audio-player'),
  playPauseBtn: document.getElementById('play-pause-btn'),
  timeCurrent: document.getElementById('time-current'),
  timeDuration: document.getElementById('time-duration'),
  progressContainer: document.getElementById('player-progress-container'),
  progressFill: document.getElementById('player-progress-fill'),
  playerMarkers: document.getElementById('player-markers'),
  logPanel: document.getElementById('log-panel'),
  logTableBody: document.getElementById('log-table-body'),
  btnExportLog: document.getElementById('btn-export-log'),
  placeholderPanel: document.getElementById('placeholder-panel'),
  mathPanel: document.getElementById('math-panel'),
  vectorABox: document.getElementById('vector-a-box'),
  similarityPercentage: document.getElementById('similarity-percentage'),
  mathExplanationText: document.getElementById('math-explanation-text'),
  physicsPanel: document.getElementById('physics-panel'),
  compareWaveformCanvas: document.getElementById('compare-waveform-canvas'),
  canvasTimeInfo: document.getElementById('canvas-time-info'),
  secondTableBody: document.getElementById('second-table-body')
};

// ==========================================================================
// 1. Three.js 3D Visualizer Setup
// ==========================================================================
function init3D() {
  const width = els.webglContainer.clientWidth;
  const height = els.webglContainer.clientHeight;

  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2('#08090c', 0.012);

  camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 1000);
  camera.position.z = 7;

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  els.webglContainer.appendChild(renderer.domElement);

  // 3D Sphere Geometry deformed by audio waves
  const geometry = new THREE.IcosahedronGeometry(2.3, 4);
  const positionAttr = geometry.attributes.position;
  const originalPositions = new Float32Array(positionAttr.count * 3);
  for (let i = 0; i < positionAttr.count; i++) {
    originalPositions[i * 3] = positionAttr.getX(i);
    originalPositions[i * 3 + 1] = positionAttr.getY(i);
    originalPositions[i * 3 + 2] = positionAttr.getZ(i);
  }
  geometry.userData = { originalPositions };

  const material = new THREE.MeshBasicMaterial({
    color: 0x3b82f6,
    wireframe: true,
    transparent: true,
    opacity: 0.5,
    blending: THREE.AdditiveBlending
  });

  waveMesh = new THREE.Mesh(geometry, material);
  scene.add(waveMesh);

  // Starfield Particles
  const particleCount = 600;
  const particleGeom = new THREE.BufferGeometry();
  const particlePositions = new Float32Array(particleCount * 3);
  for (let i = 0; i < particleCount; i++) {
    particlePositions[i * 3] = (Math.random() - 0.5) * 35;
    particlePositions[i * 3 + 1] = (Math.random() - 0.5) * 35;
    particlePositions[i * 3 + 2] = (Math.random() - 0.5) * 35;
  }
  particleGeom.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
  const particleMat = new THREE.PointsMaterial({
    color: 0x00f3ff,
    size: 0.04,
    transparent: true,
    opacity: 0.4
  });
  particleSystem = new THREE.Points(particleGeom, particleMat);
  scene.add(particleSystem);

  // Lighting
  scene.add(new THREE.AmbientLight(0xffffff, 0.4));

  // Drag controls
  els.webglContainer.addEventListener('mousedown', (e) => {
    isDragging = true;
    previousMousePosition = { x: e.clientX, y: e.clientY };
  });

  window.addEventListener('mousemove', (e) => {
    if (!isDragging || !waveMesh) return;
    const deltaX = e.clientX - previousMousePosition.x;
    const deltaY = e.clientY - previousMousePosition.y;

    waveMesh.rotation.y += deltaX * 0.007;
    waveMesh.rotation.x += deltaY * 0.007;

    previousMousePosition = { x: e.clientX, y: e.clientY };
  });

  window.addEventListener('mouseup', () => {
    isDragging = false;
  });

  // Handle Touch for Mobile
  els.webglContainer.addEventListener('touchstart', (e) => {
    if (e.touches.length === 1) {
      isDragging = true;
      previousMousePosition = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    }
  });
  window.addEventListener('touchmove', (e) => {
    if (!isDragging || !waveMesh || e.touches.length !== 1) return;
    const deltaX = e.touches[0].clientX - previousMousePosition.x;
    const deltaY = e.touches[0].clientY - previousMousePosition.y;

    waveMesh.rotation.y += deltaX * 0.007;
    waveMesh.rotation.x += deltaY * 0.007;

    previousMousePosition = { x: e.touches[0].clientX, y: e.touches[0].clientY };
  });
  window.addEventListener('touchend', () => {
    isDragging = false;
  });

  // Handle Resize
  window.addEventListener('resize', () => {
    const w = els.webglContainer.clientWidth;
    const h = els.webglContainer.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });

  animate();
}

function animate() {
  requestAnimationFrame(animate);

  // Fetch real-time visualizer details if active
  let dataArray = new Uint8Array(0);
  let state = 'Standard';

  if (analyserNode) {
    dataArray = new Uint8Array(analyserNode.frequencyBinCount);
    analyserNode.getByteFrequencyData(dataArray);

    // Identify current state during playback/recording
    if (els.audioPlayer && !els.audioPlayer.paused) {
      const t = els.audioPlayer.currentTime;
      const nameEl = document.getElementById('player-track-name');
      const activeFileName = nameEl ? nameEl.textContent : '';
      const activeFile = analyzedFiles.find(f => f.name === activeFileName);
      if (activeFile) {
        const currentBlock = activeFile.results.blocks.find(b => t >= b.start && t < b.end);
        if (currentBlock) {
          state = currentBlock.state;
        }
      } else {
        const currentBlock = currentAnalysis.blocks.find(b => t >= b.start && t < b.end);
        if (currentBlock) {
          state = currentBlock.state;
        }
      }
    } else if (mediaRecorder && mediaRecorder.state === 'recording') {
      state = els.statusIndicator.textContent.includes('Clipping') ? 'Clipping' : 
              els.statusIndicator.textContent.includes('Dead Air') ? 'Dead Air' :
              els.statusIndicator.textContent.includes('Studio') ? 'Studio' :
              els.statusIndicator.textContent.includes('Too Quiet') ? 'Too Quiet' : 'Standard';
    }
  }

  update3DVisuals(dataArray, state);
  renderer.render(scene, camera);
}

function update3DVisuals(freqData, state) {
  if (!waveMesh) return;

  let hexColor = '#3b82f6';
  let speedMultiplier = 1.0;
  let noiseIntensity = 0.25;

  switch (state) {
    case 'Clipping':
      hexColor = '#ff2d55';
      speedMultiplier = 3.8;
      noiseIntensity = 1.1;
      break;
    case 'Studio':
      hexColor = '#10b981';
      speedMultiplier = 1.5;
      noiseIntensity = 0.45;
      break;
    case 'Standard':
      hexColor = '#f59e0b';
      speedMultiplier = 1.0;
      noiseIntensity = 0.3;
      break;
    case 'Too Quiet':
      hexColor = '#6366f1';
      speedMultiplier = 0.4;
      noiseIntensity = 0.08;
      break;
    case 'Dead Air':
      hexColor = '#4b5563';
      speedMultiplier = 0.08;
      noiseIntensity = 0.01;
      break;
  }

  targetColor.set(hexColor);
  currentColor.lerp(targetColor, 0.08);
  waveMesh.material.color.copy(currentColor);

  // Animate Sphere Vertices
  const time = performance.now() * 0.001 * speedMultiplier;
  const geometry = waveMesh.geometry;
  const positionAttr = geometry.attributes.position;
  const original = geometry.userData.originalPositions;

  for (let i = 0; i < positionAttr.count; i++) {
    const ox = original[i * 3];
    const oy = original[i * 3 + 1];
    const oz = original[i * 3 + 2];

    const len = Math.sqrt(ox * ox + oy * oy + oz * oz);
    const nx = ox / len;
    const ny = oy / len;
    const nz = oz / len;

    // Displace vertex based on noise + frequency data
    const freqIndex = i % (freqData.length || 1);
    const freqRatio = freqData.length ? freqData[freqIndex] / 255.0 : 0.0;

    // Combine sine waves + sound frequencies
    const offset = Math.sin(ox * 2 + time * 3) * Math.cos(oy * 2.5 + time * 2.5) * noiseIntensity
                 + freqRatio * (noiseIntensity + 0.35);

    positionAttr.setXYZ(i, ox + nx * offset, oy + ny * offset, oz + nz * offset);
  }

  positionAttr.needsUpdate = true;

  // Auto Rotation
  if (!isDragging) {
    waveMesh.rotation.y += 0.004 * speedMultiplier;
    waveMesh.rotation.x += 0.0025 * speedMultiplier;
  }

  // Slow particle orbit
  particleSystem.rotation.y -= 0.0004;
}

// ==========================================================================
// 2. Audio Processing Infrastructure
// ==========================================================================
function initAudio() {
  if (window.__audioContext) {
    audioContext = window.__audioContext;
    analyserNode = window.__analyserNode;
    audioSourceNode = window.__audioSourceNode;
    return;
  }

  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  analyserNode = audioContext.createAnalyser();
  analyserNode.fftSize = 256;
  
  audioSourceNode = audioContext.createMediaElementSource(els.audioPlayer);
  audioSourceNode.connect(analyserNode);
  analyserNode.connect(audioContext.destination);

  // Cache globally to survive HMR reloads
  window.__audioContext = audioContext;
  window.__analyserNode = analyserNode;
  window.__audioSourceNode = audioSourceNode;
}

// Helper: Convert RMS to dBFS
function rmsToDbfs(rms) {
  if (rms <= 0) return -100;
  return 20 * Math.log10(rms);
}

// Helper: Classify decibel level
function classifyState(dbfs, peak) {
  if (peak >= -0.1) {
    return 'Clipping';
  }
  if (dbfs >= -18 && dbfs <= -12) {
    return 'Studio';
  }
  if (dbfs >= -24 && dbfs <= -6) {
    return 'Standard';
  }
  if (dbfs >= -45 && dbfs <= -30) {
    return 'Too Quiet';
  }
  if (dbfs < -50) {
    return 'Dead Air';
  }
  
  // Gaps interpolators
  if (dbfs > -6 && dbfs < -0.1) return 'Standard';
  if (dbfs > -30 && dbfs < -24) return 'Standard'; // treat slightly quiet as safe
  if (dbfs >= -50 && dbfs < -45) return 'Too Quiet';
  
  return 'Standard';
}

// Helper: Format Time in MM:SS.t
function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 10);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms}`;
}

// Helper: Format Time in MM:SS (for simple player display)
function formatTimeSimple(seconds) {
  if (isNaN(seconds)) return '0:00';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// ==========================================================================
// Autocorrelation Pitch Detector (maps pitch accurately from 60 Hz to 350 Hz)
function detectPitchAutocorrelation(channelData, startIdx, endIdx, sampleRate) {
  const maxWindowSize = 2048;
  const blockSize = endIdx - startIdx;
  const windowSize = Math.min(blockSize, maxWindowSize);
  
  const minFreq = 60;
  const maxFreq = 350;
  const maxPeriod = Math.floor(sampleRate / minFreq);
  const minPeriod = Math.floor(sampleRate / maxFreq);
  
  // Calculate RMS of this window first to make sure it's voiced
  let sqSum = 0;
  for (let i = 0; i < windowSize; i++) {
    const val = channelData[startIdx + i];
    sqSum += val * val;
  }
  const rms = Math.sqrt(sqSum / windowSize);
  const rmsDb = 20 * Math.log10(rms + 0.000001);
  
  // If it's too quiet (below -40 dBFS), it's unvoiced/silent
  if (rmsDb < -40) {
    return 0;
  }
  
  let bestLag = -1;
  let bestR = -1;
  
  // Autocorrelation
  for (let lag = minPeriod; lag <= maxPeriod; lag++) {
    let sum = 0;
    let sumSqA = 0;
    let sumSqB = 0;
    
    const limit = windowSize - lag;
    if (limit <= 0) continue;
    
    // To speed up, we step by 2 in the loop
    for (let i = 0; i < limit; i += 2) {
      const valA = channelData[startIdx + i];
      const valB = channelData[startIdx + i + lag];
      sum += valA * valB;
      sumSqA += valA * valA;
      sumSqB += valB * valB;
    }
    
    const norm = Math.sqrt(sumSqA * sumSqB);
    const r = norm > 0 ? (sum / norm) : 0;
    
    if (r > bestR) {
      bestR = r;
      bestLag = lag;
    }
  }
  
  // Only accept if correlation is high enough (e.g. > 0.45)
  if (bestR > 0.45 && bestLag > 0) {
    const pitch = sampleRate / bestLag;
    if (pitch >= minFreq && pitch <= maxFreq) {
      return pitch;
    }
  }
  
  return 0;
}

// ==========================================================================
// 3. Static File Analyzer
// ==========================================================================
async function analyzeAudioBuffer(audioBuffer) {
  const sampleRate = audioBuffer.sampleRate;
  const totalDuration = audioBuffer.duration;
  const channelData = audioBuffer.getChannelData(0); // analyze first channel
  
  const blockDuration = 0.1; // 100ms blocks
  const blockSize = Math.floor(sampleRate * blockDuration);
  const totalBlocks = Math.floor(channelData.length / blockSize);
  
  const blocks = [];
  let peakMax = -100;
  let rmsSum = 0;
  let validRmsCount = 0;
  let clippingCount = 0;
  let deadAirSeconds = 0;
  
  for (let i = 0; i < totalBlocks; i++) {
    const startIdx = i * blockSize;
    const endIdx = startIdx + blockSize;
    
    let blockPeakVal = 0;
    let blockSqSum = 0;
    let zeroCrossings = 0;
    let diffSqSum = 0;
    
    for (let s = startIdx; s < endIdx; s++) {
      const sampleVal = Math.abs(channelData[s]);
      if (sampleVal > blockPeakVal) blockPeakVal = sampleVal;
      blockSqSum += channelData[s] * channelData[s];
      
      if (s > startIdx) {
        if ((channelData[s] >= 0 && channelData[s - 1] < 0) || (channelData[s] < 0 && channelData[s - 1] >= 0)) {
          zeroCrossings++;
        }
        const diff = channelData[s] - channelData[s - 1];
        diffSqSum += diff * diff;
      }
    }
    
    const blockRmsVal = Math.sqrt(blockSqSum / blockSize);
    const blockPeakDb = rmsToDbfs(blockPeakVal);
    const blockRmsDb = rmsToDbfs(blockRmsVal);
    const zcr = zeroCrossings / blockSize;
    const diffRms = Math.sqrt(diffSqSum / blockSize);
    const hfRatio = diffRms / (blockRmsVal + 0.001);
    
    const pitchHz = detectPitchAutocorrelation(channelData, startIdx, endIdx, sampleRate);


    const blockState = classifyState(blockRmsDb, blockPeakDb);
    
    // Accumulate Peak & Average statistics
    if (blockPeakDb > peakMax) peakMax = blockPeakDb;
    if (blockRmsDb > -80) { // filter noise floor from true average
      rmsSum += blockRmsDb;
      validRmsCount++;
    }
    
    if (blockState === 'Clipping') clippingCount++;
    if (blockState === 'Dead Air') deadAirSeconds += blockDuration;
    
    blocks.push({
      index: i,
      start: i * blockDuration,
      end: (i + 1) * blockDuration,
      peak: blockPeakDb,
      rms: blockRmsDb,
      state: blockState,
      zcr: zcr,
      hfRatio: hfRatio,
      pitchHz: pitchHz
    });
  }
  
  // Smooth pitch curve with moving average filter
  for (let i = 1; i < blocks.length - 1; i++) {
    if (blocks[i].pitchHz > 0 && blocks[i-1].pitchHz > 0 && blocks[i+1].pitchHz > 0) {
      blocks[i].pitchHz = (blocks[i-1].pitchHz + blocks[i].pitchHz + blocks[i+1].pitchHz) / 3;
    }
  }
  
  const finalPeakDb = peakMax;
  const finalAvgDb = validRmsCount > 0 ? (rmsSum / validRmsCount) : -100;
  
  // Calculate Score Deductions
  let score = 100;
  
  // Clipping deduction: Proportional deduction (e.g. 25% clipping ratio = 100% deduction)
  const clippingDeduction = totalBlocks > 0 ? (clippingCount / totalBlocks) * 400 : 0;
  
  // Too Quiet deduction: Proportional deduction (e.g. 100% quiet ratio = 100% deduction)
  const tooQuietCount = blocks.filter(b => b.state === 'Too Quiet').length;
  const tooQuietDeduction = totalBlocks > 0 ? (tooQuietCount / totalBlocks) * 100 : 0;
  
  // Dead Air deduction: Proportional deduction (e.g. 33.3% dead air ratio = 100% deduction)
  const deadAirDeduction = totalDuration > 0 ? (deadAirSeconds / totalDuration) * 300 : 0;
  
  score = Math.round(score - (clippingDeduction + tooQuietDeduction + deadAirDeduction));
  score = Math.max(0, Math.min(100, score));
  
  // Assemble Error Logs (merge adjacent blocks of the same abnormal state)
  const errors = [];
  let currentErr = null;
  
  for (let i = 0; i < blocks.length; i++) {
    const b = blocks[i];
    const isAnomalous = b.state === 'Clipping' || b.state === 'Too Quiet' || b.state === 'Dead Air';
    
    if (isAnomalous) {
      if (currentErr && currentErr.state === b.state) {
        // Extend current error interval
        currentErr.end = b.end;
        if (b.peak > currentErr.peak) currentErr.peak = b.peak;
        currentErr.rmsSum += b.rms;
        currentErr.blockCount++;
      } else {
        // Close previous error if open
        if (currentErr) {
          currentErr.rms = currentErr.rmsSum / currentErr.blockCount;
          errors.push(currentErr);
        }
        // Start new error interval
        currentErr = {
          state: b.state,
          start: b.start,
          end: b.end,
          peak: b.peak,
          rmsSum: b.rms,
          blockCount: 1
        };
      }
    } else {
      if (currentErr) {
        currentErr.rms = currentErr.rmsSum / currentErr.blockCount;
        errors.push(currentErr);
        currentErr = null;
      }
    }
  }
  
  // Close any final error
  if (currentErr) {
    currentErr.rms = currentErr.rmsSum / currentErr.blockCount;
    errors.push(currentErr);
  }
  
  return {
    score,
    peakDbfs: finalPeakDb,
    avgDbfs: finalAvgDb,
    clippingBlocks: clippingCount,
    deadAirDuration: deadAirSeconds,
    duration_seconds: totalDuration,
    blocks,
    errors
  };
}

// ==========================================================================
// 4. Live Recording Engine
// ==========================================================================
async function startLiveRecording() {
  initAudio();
  if (audioContext.state === 'suspended') {
    await audioContext.resume();
  }
  
  recordedChunks = [];
  liveBlocks = [];
  recordStartTime = performance.now();
  
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    
    // Create node for live stream
    micSourceNode = audioContext.createMediaStreamSource(stream);
    micSourceNode.connect(analyserNode);
    
    // Start Recorder
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };
    
    mediaRecorder.onstop = async () => {
      els.statusIndicator.textContent = 'Engine Idle';
      els.statusIndicator.className = 'status-indicator idle';
      
      const audioBlob = new Blob(recordedChunks, { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(audioBlob);
      
      // Decode and analyze full recorded file
      els.statusIndicator.textContent = 'Analyzing Track...';
      els.statusIndicator.className = 'status-indicator analyzing';
      
      try {
        const arrayBuffer = await audioBlob.arrayBuffer();
        const decodedBuffer = await audioContext.decodeAudioData(arrayBuffer);
        const results = await analyzeAudioBuffer(decodedBuffer);
        const recordName = `Live Recording (${new Date().toLocaleTimeString()})`;
        displayAnalysisResults({ name: recordName }, results, audioUrl);
      } catch (err) {
        console.error('Failed to decode recording:', err);
        alert('Could not decode recorded audio.');
      }
      
      els.statusIndicator.textContent = 'Engine Idle';
      els.statusIndicator.className = 'status-indicator idle';
    };
    
    mediaRecorder.start();
    
    els.statusIndicator.textContent = 'Recording Live...';
    els.statusIndicator.className = 'status-indicator recording';
    
    // UI Updates
    els.startRecordBtn.classList.add('hidden');
    els.stopRecordBtn.classList.remove('hidden');
    document.querySelector('.record-pulse-container').classList.add('recording');
    els.recordStatusText.textContent = 'Recording input stream...';
    
    // Start Recording Timer & Realtime stats scan
    runRecordingLoop();
    
  } catch (err) {
    console.error('Failed to acquire microhpone:', err);
    alert('Failed to access microphone. Please check permissions.');
  }
}

function runRecordingLoop() {
  if (!mediaRecorder || mediaRecorder.state !== 'recording') return;
  
  // Update Timer
  const elapsed = (performance.now() - recordStartTime) / 1000;
  els.recordTimer.textContent = formatTime(elapsed);
  
  // Real-time decibel check from Analyser Node
  const bufferLength = analyserNode.fftSize;
  const timeData = new Float32Array(bufferLength);
  analyserNode.getFloatTimeDomainData(timeData);
  
  let peakVal = 0;
  let sqSum = 0;
  for (let i = 0; i < bufferLength; i++) {
    const val = Math.abs(timeData[i]);
    if (val > peakVal) peakVal = val;
    sqSum += timeData[i] * timeData[i];
  }
  
  const rmsVal = Math.sqrt(sqSum / bufferLength);
  const peakDb = rmsToDbfs(peakVal);
  const rmsDb = rmsToDbfs(rmsVal);
  
  const liveState = classifyState(rmsDb, peakDb);
  els.statusIndicator.textContent = `Live Recording - Level: ${Math.round(rmsDb)} dB (${liveState})`;
  
  // Add live blocks to trigger 3D color morph
  liveBlocks.push({ peak: peakDb, rms: rmsDb, state: liveState });
  
  // Update Quality Score & Stats widget in real-time during recording
  const runningStats = calculateStatsFromBlocks(liveBlocks);
  els.scorePanel.classList.remove('hidden');
  updateScoreAndStatsWidget(runningStats);
  
  recordTimerInterval = setTimeout(runRecordingLoop, 100);
}

function stopLiveRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    // Stop tracks
    micSourceNode.mediaStream.getTracks().forEach(track => track.stop());
    clearTimeout(recordTimerInterval);
    
    els.startRecordBtn.classList.remove('hidden');
    els.stopRecordBtn.classList.add('hidden');
    document.querySelector('.record-pulse-container').classList.remove('recording');
    els.recordStatusText.textContent = 'Processing final quality logs...';
  }
}

// ==========================================================================
// 5. Diagnostics Dashboard Renderer
// ==========================================================================
function displayAnalysisResults(file, results, objectUrl) {
  // Add to analyzedFiles list
  const fileEntry = {
    name: file.name,
    url: objectUrl,
    results: results
  };
  
  const existingIdx = analyzedFiles.findIndex(f => f.name === file.name);
  if (existingIdx !== -1) {
    analyzedFiles[existingIdx] = fileEntry;
  } else {
    analyzedFiles.push(fileEntry);
  }
  
  // Set currentAnalysis to the latest results for fallback compatibility
  currentAnalysis = results;
  
  // Show panels & hide placeholders
  els.placeholderPanel.classList.add('hidden');
  els.scorePanel.classList.remove('hidden');
  els.playerPanel.classList.remove('hidden');
  els.logPanel.classList.remove('hidden');
  
  // Load into HTML audio player
  els.audioPlayer.src = objectUrl;
  els.audioPlayer.load();
  const nameEl = document.getElementById('player-track-name');
  if (nameEl) nameEl.textContent = file.name;
  
  // Update score & stats widget
  updateScoreAndStatsWidget(results);
  
  // Re-generate timeline markers
  generateTimelineMarkers(results);
  
  // Update log table (all files combined)
  updateLogTable();
}

function switchActiveFile(fileName) {
  const fileEntry = analyzedFiles.find(f => f.name === fileName);
  if (!fileEntry) return;

  // Update currentAnalysis reference
  currentAnalysis = fileEntry.results;

  // Load in player if not active
  if (els.audioPlayer.src !== fileEntry.url) {
    els.audioPlayer.src = fileEntry.url;
    els.audioPlayer.load();
    const nameEl = document.getElementById('player-track-name');
    if (nameEl) nameEl.textContent = fileEntry.name;
  }

  // Update score & stats widget
  updateScoreAndStatsWidget(fileEntry.results);
  
  // Re-generate timeline markers
  generateTimelineMarkers(fileEntry.results);
}

function calculateStatsFromBlocks(blocks, blockDuration = 0.1) {
  if (!blocks || blocks.length === 0) {
    return {
      score: 100,
      peakDbfs: -100,
      avgDbfs: -100,
      clippingBlocks: 0,
      deadAirDuration: 0
    };
  }
  
  let peakMax = -100;
  let rmsSum = 0;
  let validRmsCount = 0;
  let clippingCount = 0;
  let deadAirSeconds = 0;
  
  blocks.forEach(b => {
    if (b.peak > peakMax) peakMax = b.peak;
    if (b.rms > -80) {
      rmsSum += b.rms;
      validRmsCount++;
    }
    if (b.state === 'Clipping') clippingCount++;
    if (b.state === 'Dead Air') deadAirSeconds += blockDuration;
  });
  
  const finalPeakDb = peakMax;
  const finalAvgDb = validRmsCount > 0 ? (rmsSum / validRmsCount) : -100;
  
  // Deductions (proportional to segment length)
  let score = 100;
  const totalBlocks = blocks.length;
  const totalDuration = totalBlocks * blockDuration;
  
  const clippingDeduction = totalBlocks > 0 ? (clippingCount / totalBlocks) * 400 : 0;
  const tooQuietCount = blocks.filter(b => b.state === 'Too Quiet').length;
  const tooQuietDeduction = totalBlocks > 0 ? (tooQuietCount / totalBlocks) * 100 : 0;
  const deadAirDeduction = totalDuration > 0 ? (deadAirSeconds / totalDuration) * 300 : 0;
  
  score = Math.round(score - (clippingDeduction + tooQuietDeduction + deadAirDeduction));
  score = Math.max(0, Math.min(100, score));
  
  return {
    score,
    peakDbfs: finalPeakDb,
    avgDbfs: finalAvgDb,
    clippingBlocks: clippingCount,
    deadAirDuration: deadAirSeconds
  };
}

function updateScoreAndStatsWidget(results) {
  // --- Vector A Calculations ---
  const blocks = results.blocks || [];
  let sumRms = 0;
  let sumZcr = 0;
  let sumHf = 0;
  
  blocks.forEach(b => {
    sumRms += b.rms;
    sumZcr += b.zcr || 0;
    sumHf += b.hfRatio || 0;
  });
  
  const avgRms = blocks.length > 0 ? (sumRms / blocks.length) : -100;
  const avgZcr = blocks.length > 0 ? (sumZcr / blocks.length) : 0;
  const avgHf = blocks.length > 0 ? (sumHf / blocks.length) : 0;
  
  // Calculate ZCR variance
  let sumZcrVar = 0;
  blocks.forEach(b => {
    const diff = (b.zcr || 0) - avgZcr;
    sumZcrVar += diff * diff;
  });
  const zcrVar = blocks.length > 0 ? (sumZcrVar / blocks.length) : 0;
  
  // Normalize parameters to [0.0, 1.0]
  const normRms = Math.max(0, Math.min(1.0, (avgRms + 60) / 60));
  const normPeak = Math.max(0, Math.min(1.0, (results.peakDbfs + 60) / 60));
  const normZcr = Math.max(0, Math.min(1.0, avgZcr * 6.0)); 
  const normZcrVar = Math.max(0, Math.min(1.0, zcrVar * 80.0));
  const normHf = Math.max(0, Math.min(1.0, avgHf / 2.5));
  
  const vecA = [
    parseFloat(normRms.toFixed(2)),
    parseFloat(normPeak.toFixed(2)),
    parseFloat(normZcr.toFixed(2)),
    parseFloat(normZcrVar.toFixed(2)),
    parseFloat(normHf.toFixed(2))
  ];
  
  // Reference Vector B
  const vecB = [0.70, 0.85, 0.40, 0.15, 0.35];
  
  // Cosine Similarity: (A . B) / (||A|| * ||B||)
  let dotProduct = 0;
  let magASq = 0;
  let magBSq = 0;
  
  for (let i = 0; i < 5; i++) {
    dotProduct += vecA[i] * vecB[i];
    magASq += vecA[i] * vecA[i];
    magBSq += vecB[i] * vecB[i];
  }
  
  const magA = Math.sqrt(magASq);
  const magB = Math.sqrt(magBSq);
  
  let similarity = 0;
  if (magA > 0 && magB > 0) {
    similarity = dotProduct / (magA * magB);
  }
  
  const similarityPct = (similarity * 100).toFixed(1);
  
  // Update Math UI Elements
  if (els.mathPanel) {
    els.mathPanel.classList.remove('hidden');
  }
  if (els.vectorABox) {
    els.vectorABox.textContent = `[${vecA.map(v => v.toFixed(2)).join(', ')}]`;
  }
  if (els.similarityPercentage) {
    els.similarityPercentage.textContent = `${similarityPct}%`;
    if (similarity >= 0.92) {
      els.similarityPercentage.style.color = 'var(--color-studio)';
    } else if (similarity >= 0.80) {
      els.similarityPercentage.style.color = 'var(--color-standard)';
    } else {
      els.similarityPercentage.style.color = 'var(--color-clipping)';
    }
  }
  if (els.mathExplanationText) {
    let explanation = '';
    if (similarity >= 0.92) {
      explanation = `<strong>Pristine Acoustic Matching:</strong> The audio track exhibits a high cosine match index (${similarityPct}%) with reference voice patterns. Loudness levels are balanced, zero-crossing spectral pitch variance is within standard vocal dynamics, and high-frequency distortion products are minimal.`;
    } else if (similarity >= 0.80) {
      explanation = `<strong>Moderate Acoustic Matching:</strong> The similarity score is ${similarityPct}%. There are moderate variations in the pitch variance or loudness profile relative to the studio baseline, indicating potential field recording noise or mild vocal strain.`;
    } else {
      explanation = `<strong>Low Acoustic Matching (${similarityPct}%):</strong> Significant acoustic vector mismatch. This indicates either excessive room reverberation, heavy compression noise (high high-frequency ratio), monotone digital anomalies, or digital clipping distortion.`;
    }
    els.mathExplanationText.innerHTML = explanation;
  }

  // 1. Overall Score Ring animation
  els.scoreText.textContent = results.score;
  const dashOffset = 251.2 - (251.2 * results.score) / 100;
  els.scoreRing.style.strokeDashoffset = dashOffset;
  
  // Adjust progress ring color based on score grade
  if (results.score >= 90) {
    els.scoreRing.style.stroke = 'var(--color-studio)';
  } else if (results.score >= 70) {
    els.scoreRing.style.stroke = 'var(--color-standard)';
  } else if (results.score >= 45) {
    els.scoreRing.style.stroke = 'var(--color-quiet)';
  } else {
    els.scoreRing.style.stroke = 'var(--color-clipping)';
  }
  
  // 2. Statistics Card populating
  els.statPeakDb.textContent = `${results.peakDbfs.toFixed(1)} dBFS`;
  els.statAvgDb.textContent = `${results.avgDbfs.toFixed(1)} dBFS`;
  els.statClippingCount.textContent = results.clippingBlocks;
  els.statDeadAirDuration.textContent = `${results.deadAirDuration.toFixed(1)}s`;
  
  // 3. Verdict Banner
  let verdictClass = '';
  let verdictTitle = '';
  let verdictDesc = '';
  
  if (results.score >= 90) {
    verdictTitle = '🥇 EXCELLENT STUDIO FIDELITY';
    verdictDesc = 'Pristine voice parameters. Low distortion, great dynamic range, ideal for production.';
    verdictClass = '';
  } else if (results.score >= 75) {
    verdictTitle = '🥈 SAFE BROADCAST QUALITY';
    verdictDesc = 'Overall standard bounds met. Minor level inconsistencies but safe for publication.';
    verdictClass = 'warning';
  } else if (results.score >= 50) {
    verdictTitle = '⚠️ MODERATE AUDIO ARTIFACTS';
    verdictDesc = 'Frequent clipping spikes or excessive dead air segments detected. Needs normalization.';
    verdictClass = 'warning';
  } else {
    verdictTitle = '🚨 CRITICAL QUALITY FAILURE';
    verdictDesc = 'Severe clipping distortions or heavy silent intervals. Requires immediate audio re-recording.';
    verdictClass = 'danger';
  }
  
  els.verdictBanner.className = `verdict-banner ${verdictClass}`;
  els.verdictTitle.textContent = verdictTitle;
  els.verdictDesc.textContent = verdictDesc;
  
  // Update Mini Score in Bottom Bar
  const miniScoreVal = document.getElementById('mini-score-val');
  if (miniScoreVal) {
    miniScoreVal.textContent = results.score;
    if (results.score >= 90) {
      miniScoreVal.style.color = 'var(--color-studio)';
    } else if (results.score >= 70) {
      miniScoreVal.style.color = 'var(--color-standard)';
    } else if (results.score >= 45) {
      miniScoreVal.style.color = 'var(--color-quiet)';
    } else {
      miniScoreVal.style.color = 'var(--color-clipping)';
    }
  }

  // Update Player Track Status text
  const trackStatus = document.getElementById('player-track-status');
  if (trackStatus) {
    if (results.score >= 90) {
      trackStatus.textContent = 'QA Passed: Pristine';
      trackStatus.style.color = 'var(--color-studio)';
    } else if (results.score >= 70) {
      trackStatus.textContent = 'QA Passed: Safe';
      trackStatus.style.color = 'var(--color-standard)';
    } else {
      trackStatus.textContent = 'QA Failed: Anomalies';
      trackStatus.style.color = 'var(--color-clipping)';
    }
  }

  // Show physics panel
  if (els.physicsPanel) {
    els.physicsPanel.classList.remove('hidden');
  }

  // 6. Draw Comparative Waveform
  if (typeof drawComparativeWaveform === 'function') {
    drawComparativeWaveform(results);
  }

  // 7. Populate Second-by-Second Table
  if (typeof populateSecondTable === 'function') {
    populateSecondTable(results);
  }
}

function generateTimelineMarkers(results) {
  els.playerMarkers.innerHTML = '';
  const totalDur = els.audioPlayer.duration || 1; // avoid division by 0
  
  results.errors.forEach(err => {
    const leftPercent = (err.start / totalDur) * 100;
    const widthPercent = ((err.end - err.start) / totalDur) * 100;
    
    const marker = document.createElement('div');
    marker.className = `time-marker mark-${err.state.toLowerCase().replace(' ', '-')}`;
    marker.style.left = `${leftPercent}%`;
    marker.style.width = `${Math.max(0.5, widthPercent)}%`; // ensure it remains visible
    
    els.playerMarkers.appendChild(marker);
  });
}

function updateLogTable() {
  els.logTableBody.innerHTML = '';
  
  // Accumulate all errors
  let allErrors = [];
  analyzedFiles.forEach(fileEntry => {
    fileEntry.results.errors.forEach(err => {
      allErrors.push({
        fileName: fileEntry.name,
        err: err
      });
    });
  });
  
  if (allErrors.length === 0) {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td colspan="6" style="text-align: center; color: var(--color-studio); font-weight: 600; padding: 24px;">
        ✨ Pristine Pass! No clipping, silent gaps, or under-amplified blocks detected.
      </td>
    `;
    els.logTableBody.appendChild(row);
    return;
  }
  
  allErrors.forEach((item, idx) => {
    const row = document.createElement('tr');
    row.id = `log-row-${idx}`;
    row.dataset.fileName = item.fileName;
    row.dataset.start = item.err.start;
    row.dataset.end = item.err.end;
    
    const timeStr = `${item.err.start.toFixed(1)}s - ${item.err.end.toFixed(1)}s`;
    const stateBadge = `<span class="badge badge-${item.err.state.toLowerCase().replace(' ', '-')}">${item.err.state}</span>`;
    
    let description = '';
    switch (item.err.state) {
      case 'Clipping':
        description = '💥 <strong>Volume Overload:</strong> Flat curves. Causes harsh, buzzing distortion.';
        break;
      case 'Too Quiet':
        description = '📉 <strong>Under-amplified:</strong> Voice buried near floor hum. Raising this adds heavy hiss.';
        break;
      case 'Dead Air':
        description = '🔇 <strong>Digital Silence:</strong> Complete absence of sound energy. Sounds like drop-out.';
        break;
    }
    
    row.innerHTML = `
      <td style="font-weight: 600; color: var(--accent-cyan);">${item.fileName}</td>
      <td>${timeStr}</td>
      <td>${stateBadge}</td>
      <td>${item.err.peak.toFixed(1)} dBFS</td>
      <td><span class="table-desc">${description}</span></td>
      <td><button class="play-row-btn" data-file="${item.fileName}" data-start="${item.err.start}">Seek & Play</button></td>
    `;
    
    els.logTableBody.appendChild(row);
  });
  
  // Attach listeners to rows play button
  document.querySelectorAll('.play-row-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const fileName = e.target.dataset.file;
      const startT = parseFloat(e.target.dataset.start);
      
      switchActiveFile(fileName);
      els.audioPlayer.currentTime = startT;
      els.audioPlayer.play();
      els.playPauseBtn.textContent = '⏸';
    });
  });
}

// ==========================================================================
// 6. Timeline and Audio Elements Bindings
// ==========================================================================
function bindAudioEvents() {
  els.audioPlayer.addEventListener('loadedmetadata', () => {
    els.timeDuration.textContent = formatTimeSimple(els.audioPlayer.duration);
    els.timeCurrent.textContent = formatTimeSimple(0);
  });
  
  els.audioPlayer.addEventListener('timeupdate', () => {
    const cur = els.audioPlayer.currentTime;
    const dur = els.audioPlayer.duration || 1;
    els.timeCurrent.textContent = formatTimeSimple(cur);
    els.progressFill.style.width = `${(cur / dur) * 100}%`;
    
    // Highlight matching error row in table
    highlightLogTableRow(cur);
    
    // Dynamically update stats cards during playback
    if (els.audioPlayer && !els.audioPlayer.paused && cur > 0) {
      const nameEl = document.getElementById('player-track-name');
      const activeFileName = nameEl ? nameEl.textContent : '';
      const activeFile = analyzedFiles.find(f => f.name === activeFileName);
      if (activeFile) {
        const playedBlocks = activeFile.results.blocks.filter(b => b.start <= cur);
        if (playedBlocks.length > 0) {
          const runningStats = calculateStatsFromBlocks(playedBlocks);
          updateScoreAndStatsWidget(runningStats);
        }
      }
    }

    // Update playhead on comparative waveform canvas
    if (typeof drawPlayheadOnCanvas === 'function') {
      drawPlayheadOnCanvas(cur, dur);
    }
    
    // Highlight second-by-second table row
    if (typeof highlightSecondTableRow === 'function') {
      highlightSecondTableRow(cur);
    }
    
    // Update text time indicator above canvas
    if (els.canvasTimeInfo) {
      els.canvasTimeInfo.textContent = `${formatTime(cur)} / ${formatTime(dur)} (Click waveform to seek)`;
    }
  });
  
  els.audioPlayer.addEventListener('ended', () => {
    els.playPauseBtn.textContent = '▶';
    // Restore overall stats of the active file
    const nameEl = document.getElementById('player-track-name');
    const activeFileName = nameEl ? nameEl.textContent : '';
    const activeFile = analyzedFiles.find(f => f.name === activeFileName);
    if (activeFile) {
      updateScoreAndStatsWidget(activeFile.results);
    }
  });
  
  els.playPauseBtn.addEventListener('click', () => {
    initAudio();
    if (audioContext.state === 'suspended') {
      audioContext.resume();
    }
    
    if (els.audioPlayer.paused) {
      els.audioPlayer.play();
      els.playPauseBtn.textContent = '⏸';
    } else {
      els.audioPlayer.pause();
      els.playPauseBtn.textContent = '▶';
    }
  });
  
  els.progressContainer.addEventListener('click', (e) => {
    const rect = els.progressContainer.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const width = rect.width;
    const seekRatio = clickX / width;
    
    els.audioPlayer.currentTime = seekRatio * els.audioPlayer.duration;
  });

  // comparative canvas click seeking
  if (els.compareWaveformCanvas) {
    els.compareWaveformCanvas.addEventListener('click', (e) => {
      const rect = els.compareWaveformCanvas.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const width = rect.width;
      
      const chartWidth = width - chartPadding.left - chartPadding.right;
      let relativeX = clickX - chartPadding.left;
      if (relativeX < 0) relativeX = 0;
      if (relativeX > chartWidth) relativeX = chartWidth;
      
      const seekRatio = relativeX / chartWidth;
      
      if (els.audioPlayer.duration) {
        els.audioPlayer.currentTime = seekRatio * els.audioPlayer.duration;
      }
    });
  }
}

function highlightLogTableRow(currentTime) {
  const nameEl = document.getElementById('player-track-name');
  const activeFileName = nameEl ? nameEl.textContent : '';

  const rows = els.logTableBody.querySelectorAll('tr');
  rows.forEach(row => {
    const rowFile = row.dataset.fileName;
    const start = parseFloat(row.dataset.start);
    const end = parseFloat(row.dataset.end);
    
    if (rowFile === activeFileName && currentTime >= start && currentTime < end) {
      row.classList.add('active-row');
    } else {
      row.classList.remove('active-row');
    }
  });
}

// Export Log to CSV
function exportLogToCSV() {
  let allErrors = [];
  analyzedFiles.forEach(fileEntry => {
    fileEntry.results.errors.forEach(err => {
      allErrors.push({
        fileName: fileEntry.name,
        err: err
      });
    });
  });

  if (!allErrors.length) return;
  
  let csvContent = 'data:text/csv;charset=utf-8,';
  csvContent += 'File Name,Start Time (s),End Time (s),Error Type,Peak dBFS,Average dBFS,Description\r\n';
  
  allErrors.forEach(item => {
    const startStr = item.err.start.toFixed(2);
    const endStr = item.err.end.toFixed(2);
    const description = item.err.state === 'Clipping' ? 'Volume Overload' : item.err.state === 'Too Quiet' ? 'Under-amplified' : 'Digital Silence';
    
    csvContent += `"${item.fileName}",${startStr},${endStr},"${item.err.state}",${item.err.peak.toFixed(2)},${item.err.rmsSum / item.err.blockCount},"${description}"\r\n`;
  });
  
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `clipcheck_combined_qa_log.csv`);
  document.body.appendChild(link);
  
  link.click();
  document.body.removeChild(link);
}

// ==========================================================================
// 7. Tab toggles & Drag-Drop Interfaces
// ==========================================================================
function bindUIEvents() {
  // Tabs Toggle
  els.tabUpload.addEventListener('click', () => {
    els.tabUpload.classList.add('active');
    els.tabRecord.classList.remove('active');
    els.contentUpload.classList.add('active');
    els.contentRecord.classList.remove('active');
  });
  
  els.tabRecord.addEventListener('click', () => {
    els.tabRecord.classList.add('active');
    els.tabUpload.classList.remove('active');
    els.contentRecord.classList.add('active');
    els.contentUpload.classList.remove('active');
  });
  
  // Click drop zone to trigger file picker
  els.dropZone.addEventListener('click', (e) => {
    if (e.target !== els.browseBtn) {
      els.fileInput.click();
    }
  });

  // Global window drag & drop prevention to stop file navigation
  window.addEventListener('dragover', (e) => e.preventDefault(), false);
  window.addEventListener('drop', (e) => e.preventDefault(), false);

  // Drag & Drop
  els.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    els.dropZone.classList.add('dragover');
  });
  
  els.dropZone.addEventListener('dragleave', () => {
    els.dropZone.classList.remove('dragover');
  });
  
  els.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    els.dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });
  
  els.browseBtn.addEventListener('click', (e) => {
    e.stopPropagation(); // stop triggering zone click again
    els.fileInput.click();
  });
  
  els.fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });
  
  els.analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    
    initAudio();
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }
    
    els.statusIndicator.textContent = 'Analyzing Track...';
    els.statusIndicator.className = 'status-indicator analyzing';
    
    const fileReader = new FileReader();
    fileReader.onload = async (e) => {
      const arrayBuffer = e.target.result;
      try {
        const decodedBuffer = await audioContext.decodeAudioData(arrayBuffer);
        const results = await analyzeAudioBuffer(decodedBuffer);
        
        // Load in HTML audio player
        const objectUrl = URL.createObjectURL(currentFile);
        
        displayAnalysisResults(currentFile, results, objectUrl);
      } catch (err) {
        console.error('Offline audio decode failed:', err);
        alert('Decoding failed. Please upload a standard audio file (MP3, WAV).');
      }
      
      els.statusIndicator.textContent = 'Engine Idle';
      els.statusIndicator.className = 'status-indicator idle';
    };
    fileReader.readAsArrayBuffer(currentFile);
  });
  
  // Microphone Event bindings
  els.startRecordBtn.addEventListener('click', startLiveRecording);
  els.stopRecordBtn.addEventListener('click', stopLiveRecording);
  
  // CSV Export
  els.btnExportLog.addEventListener('click', exportLogToCSV);
  
  const btnExportBottom = document.getElementById('btn-export-log-bottom');
  if (btnExportBottom) {
    btnExportBottom.addEventListener('click', exportLogToCSV);
  }
}

function handleSelectedFile(file) {
  const allowedExtensions = ['mp3', 'wav', 'ogg', 'm4a', 'aac', 'flac', 'webm', 'wma', 'mp4', 'mkv', 'mov', '3gp'];
  const fileExtension = file.name.split('.').pop().toLowerCase();
  const isAudioOrVideoMime = file.type.startsWith('audio/') || file.type.startsWith('video/');
  const isAudioOrVideoExt = allowedExtensions.includes(fileExtension);
  
  if (!isAudioOrVideoMime && !isAudioOrVideoExt) {
    alert('Please upload a valid audio or video file (MP3, WAV, OGG, M4A, FLAC, MP4, etc.).');
    return;
  }
  currentFile = file;
  
  const nameEl = document.getElementById('player-track-name');
  if (nameEl) nameEl.textContent = file.name;
  
  els.fileName.textContent = file.name;
  
  const sizeMb = file.size / (1024 * 1024);
  els.fileSize.textContent = `${sizeMb.toFixed(2)} MB`;
  
  els.fileInfoContainer.classList.remove('hidden');
}

// ==========================================================================
// 7B. Acoustic Physics & Waveform Comparison Functions
// ==========================================================================
let waveformPoints = []; // caches values for redraws
const chartPadding = { left: 65, right: 30, top: 20, bottom: 40 };

// Helper: Map Pitch Hz to Canvas Y coordinates (padded chart area)
function hzToY(hz, height) {
  const chartHeight = height - chartPadding.top - chartPadding.bottom;
  if (hz <= 0) return chartPadding.top + chartHeight; // bottom of the chart
  const clampedHz = Math.max(50, Math.min(350, hz));
  const ratio = (clampedHz - 50) / 300;
  return chartPadding.top + chartHeight * (1 - ratio);
}

// Helper: Map Time seconds to Canvas X coordinates (padded chart area)
function timeToX(time, duration, width) {
  const chartWidth = width - chartPadding.left - chartPadding.right;
  const ratio = duration > 0 ? time / duration : 0;
  return chartPadding.left + ratio * chartWidth;
}

// Draw chart axes, tick marks, and grids
function drawAxesAndGrid(ctx, width, height, duration) {
  const chartWidth = width - chartPadding.left - chartPadding.right;
  const chartHeight = height - chartPadding.top - chartPadding.bottom;
  
  // 1. Draw grid lines (horizontal)
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
  ctx.lineWidth = 1;
  const gridLines = 8;
  for (let i = 1; i < gridLines; i++) {
    const y = chartPadding.top + (chartHeight / gridLines) * i;
    ctx.beginPath();
    ctx.moveTo(chartPadding.left, y);
    ctx.lineTo(chartPadding.left + chartWidth, y);
    ctx.stroke();
  }
  
  // 2. Draw Y-axis line (vertical)
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(chartPadding.left, chartPadding.top);
  ctx.lineTo(chartPadding.left, chartPadding.top + chartHeight);
  ctx.stroke();
  
  // Draw X-axis line (horizontal)
  ctx.beginPath();
  ctx.moveTo(chartPadding.left, chartPadding.top + chartHeight);
  ctx.lineTo(chartPadding.left + chartWidth, chartPadding.top + chartHeight);
  ctx.stroke();
  
  // 3. Draw Y-axis tick marks & labels
  const yTicks = [
    { hz: 50, label: '50 Hz' },
    { hz: 80, label: '80 Hz (Low)' },
    { hz: 140, label: '140 Hz (Normal)' },
    { hz: 240, label: '240 Hz (High)' },
    { hz: 350, label: '350 Hz' }
  ];
  
  ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
  ctx.font = '9px JetBrains Mono';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  
  yTicks.forEach(tick => {
    const y = hzToY(tick.hz, height);
    
    // Draw tick mark
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.beginPath();
    ctx.moveTo(chartPadding.left - 5, y);
    ctx.lineTo(chartPadding.left, y);
    ctx.stroke();
    
    // Draw label
    ctx.fillText(tick.label, chartPadding.left - 8, y);
  });
  
  // 4. Draw X-axis tick marks & labels
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  
  const dur = duration || 1;
  let step = 1;
  if (dur <= 10) step = 1;
  else if (dur <= 30) step = 5;
  else if (dur <= 60) step = 10;
  else step = 20;
  
  for (let t = 0; t <= dur; t += step) {
    const x = timeToX(t, dur, width);
    
    // Draw tick mark
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.beginPath();
    ctx.moveTo(x, chartPadding.top + chartHeight);
    ctx.lineTo(x, chartPadding.top + chartHeight + 5);
    ctx.stroke();
    
    // Draw label
    ctx.fillText(`${t}s`, x, chartPadding.top + chartHeight + 8);
    
    // Draw subtle vertical grid line
    if (t > 0 && t < dur) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
      ctx.beginPath();
      ctx.moveTo(x, chartPadding.top);
      ctx.lineTo(x, chartPadding.top + chartHeight);
      ctx.stroke();
    }
  }
}

// Draw standard dashed reference lines
function drawReferenceLines(ctx, width, height) {
  const chartWidth = width - chartPadding.left - chartPadding.right;
  
  // High Standard Pitch Line (240 Hz) - Red Dashed
  const yHigh = hzToY(240, height);
  ctx.strokeStyle = 'rgba(255, 45, 85, 0.6)';
  ctx.setLineDash([3, 4]);
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.moveTo(chartPadding.left, yHigh);
  ctx.lineTo(chartPadding.left + chartWidth, yHigh);
  ctx.stroke();
  
  // Normal Standard Pitch Line (140 Hz) - Gold Dashed
  const yNorm = hzToY(140, height);
  ctx.strokeStyle = 'rgba(245, 158, 11, 0.5)';
  ctx.beginPath();
  ctx.moveTo(chartPadding.left, yNorm);
  ctx.lineTo(chartPadding.left + chartWidth, yNorm);
  ctx.stroke();
  
  // Low Standard Pitch Line (80 Hz) - Blue/Indigo Dashed
  const yLow = hzToY(80, height);
  ctx.strokeStyle = 'rgba(99, 102, 241, 0.6)';
  ctx.beginPath();
  ctx.moveTo(chartPadding.left, yLow);
  ctx.lineTo(chartPadding.left + chartWidth, yLow);
  ctx.stroke();
  
  ctx.setLineDash([]); // reset
}

function drawComparativeWaveform(results) {
  const canvas = els.compareWaveformCanvas;
  if (!canvas) return;
  
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = 280 * window.devicePixelRatio;
  
  const ctx = canvas.getContext('2d');
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  
  const width = rect.width;
  const height = 280;
  const blocks = results.blocks || [];
  waveformPoints = [];
  
  // Cache and ensure duration is computed
  if (!results.duration_seconds && els.audioPlayer && els.audioPlayer.duration) {
    results.duration_seconds = els.audioPlayer.duration;
  }
  const duration = results.duration_seconds || (blocks.length * 0.1) || 1;
  results.duration_seconds = duration;
  
  // Base background
  ctx.fillStyle = '#08090c';
  ctx.fillRect(0, 0, width, height);
  
  // Draw Axes and Grid
  drawAxesAndGrid(ctx, width, height, duration);
  
  // Draw Standard reference lines
  drawReferenceLines(ctx, width, height);
  
  if (blocks.length === 0) return;
  
  // Cache points
  for (let i = 0; i < blocks.length; i++) {
    const b = blocks[i];
    const time = b.start;
    const x = timeToX(time, duration, width);
    const y = hzToY(b.pitchHz, height);
    waveformPoints.push({ x, y, pitchHz: b.pitchHz, time });
  }
  
  // Draw User Vocal Pitch Curve (Glowing Cyan Continuous Line)
  ctx.shadowBlur = 6;
  ctx.shadowColor = 'var(--accent-cyan)';
  ctx.strokeStyle = 'var(--accent-cyan)';
  ctx.lineWidth = 4.0;
  ctx.beginPath();
  
  let activeSegment = false;
  for (let i = 0; i < waveformPoints.length; i++) {
    const pt = waveformPoints[i];
    if (pt.pitchHz > 0) {
      if (!activeSegment) {
        ctx.moveTo(pt.x, pt.y);
        activeSegment = true;
      } else {
        ctx.lineTo(pt.x, pt.y);
      }
    } else {
      activeSegment = false;
    }
  }
  ctx.stroke();
  
  // Highlight High Pitch Violations in Glowing Red (>240 Hz)
  ctx.shadowColor = 'var(--color-clipping)';
  ctx.strokeStyle = 'var(--color-clipping)';
  ctx.lineWidth = 5.0;
  ctx.beginPath();
  
  let highSegment = false;
  for (let i = 0; i < waveformPoints.length; i++) {
    const pt = waveformPoints[i];
    if (pt.pitchHz > 240) {
      if (!highSegment) {
        ctx.moveTo(pt.x, pt.y);
        highSegment = true;
      } else {
        ctx.lineTo(pt.x, pt.y);
      }
    } else {
      highSegment = false;
    }
  }
  ctx.stroke();
  ctx.shadowBlur = 0; // reset
  
  drawPlayheadOnCanvas(0, duration);
}

function drawPlayheadOnCanvas(currentTime, duration) {
  const canvas = els.compareWaveformCanvas;
  if (!canvas || waveformPoints.length === 0) return;
  
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = 280;
  const dur = currentAnalysis.duration_seconds || duration || 1;
  
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#08090c';
  ctx.fillRect(0, 0, width, height);
  
  // Draw Axes and Grid
  drawAxesAndGrid(ctx, width, height, dur);
  
  // Draw reference standard lines
  drawReferenceLines(ctx, width, height);
  
  // Re-draw continuous voice line
  ctx.strokeStyle = 'var(--accent-cyan)';
  ctx.lineWidth = 4.0;
  ctx.beginPath();
  let activeSegment = false;
  for (let i = 0; i < waveformPoints.length; i++) {
    const pt = waveformPoints[i];
    if (pt.pitchHz > 0) {
      if (!activeSegment) {
        ctx.moveTo(pt.x, pt.y);
        activeSegment = true;
      } else {
        ctx.lineTo(pt.x, pt.y);
      }
    } else {
      activeSegment = false;
    }
  }
  ctx.stroke();
  
  // Re-draw high pitch segments in red (> 240 Hz)
  ctx.strokeStyle = 'var(--color-clipping)';
  ctx.lineWidth = 5.0;
  ctx.beginPath();
  let highSegment = false;
  for (let i = 0; i < waveformPoints.length; i++) {
    const pt = waveformPoints[i];
    if (pt.pitchHz > 240) {
      if (!highSegment) {
        ctx.moveTo(pt.x, pt.y);
        highSegment = true;
      } else {
        ctx.lineTo(pt.x, pt.y);
      }
    } else {
      highSegment = false;
    }
  }
  ctx.stroke();
  
  // Draw Playhead
  const x = timeToX(currentTime, dur, width);
  const chartHeight = height - chartPadding.top - chartPadding.bottom;
  
  ctx.strokeStyle = 'var(--accent-cyan)';
  ctx.lineWidth = 1.5;
  ctx.shadowColor = 'var(--accent-cyan)';
  ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.moveTo(x, chartPadding.top);
  ctx.lineTo(x, chartPadding.top + chartHeight);
  ctx.stroke();
  ctx.shadowBlur = 0;
}

function populateSecondTable(results) {
  els.secondTableBody.innerHTML = '';
  const blocks = results.blocks || [];
  const duration = Math.floor(results.duration_seconds);
  
  for (let sec = 0; sec <= duration; sec++) {
    const secBlocks = blocks.filter(b => b.start >= sec && b.start < (sec + 1));
    if (secBlocks.length === 0) continue;
    
    let sumPitch = 0;
    let voicedCount = 0;
    let clipCount = 0;
    let deadAirCount = 0;
    let quietCount = 0;
    
    secBlocks.forEach(b => {
      if (b.pitchHz > 0) {
        sumPitch += b.pitchHz;
        voicedCount++;
      }
      if (b.state === 'Clipping') clipCount++;
      if (b.state === 'Dead Air') deadAirCount++;
      if (b.state === 'Too Quiet') quietCount++;
    });
    
    const avgPitch = voicedCount > 0 ? (sumPitch / voicedCount) : 0;
    
    // Pitch Corridor calculations (80 to 240 Hz)
    let deviation = 0;
    let devClass = 'good';
    let comment = '';
    let dominantState = 'Standard';
    
    if (avgPitch > 240) {
      deviation = avgPitch - 240;
      devClass = 'bad'; // High pitch violation
      comment = `⚠️ <strong>High Pitch:</strong> Vocal pitch hit <strong>${Math.round(avgPitch)} Hz</strong>, exceeding standard limit. Acoustic strain detected.`;
      dominantState = 'High Pitch'; // represent as high alarm
    } else if (avgPitch > 0 && avgPitch < 80) {
      deviation = avgPitch - 80;
      devClass = 'warn'; // Low pitch
      comment = `📉 <strong>Low Pitch:</strong> Vocal pitch dropped to <strong>${Math.round(avgPitch)} Hz</strong>. Monotone baseline.`;
      dominantState = 'Too Quiet';
    } else if (avgPitch === 0) {
      deviation = 0;
      devClass = 'warn';
      comment = `🔇 <strong>Unvoiced Gap:</strong> Sound drops below threshold. Dead air or silence zone.`;
      dominantState = 'Dead Air';
    } else {
      deviation = 0;
      devClass = 'good';
      comment = `🟢 <strong>Optimal Pitch:</strong> Pitch is stable at <strong>${Math.round(avgPitch)} Hz</strong>. Natural voice harmony corridor.`;
      dominantState = 'Studio';
    }
    
    const devText = deviation === 0 ? '0 Hz' : `${deviation > 0 ? '+' : ''}${Math.round(deviation)} Hz`;
    
    // Applied penalty inside this specific second
    let penalty = 0;
    if (clipCount > 0) penalty += clipCount * 4;
    if (quietCount > 0) penalty += quietCount * 1;
    if (deadAirCount > 0) penalty += 3.0;
    
    const penaltyText = penalty === 0 ? '0%' : `-${Math.round(penalty)}%`;
    const stateBadge = `<span class="badge badge-${dominantState.toLowerCase().replace(' ', '-')}">${dominantState}</span>`;
    
    const row = document.createElement('tr');
    row.id = `sec-row-${sec}`;
    row.innerHTML = `
      <td>${sec}.0s</td>
      <td>80 - 240 Hz</td>
      <td style="font-family: var(--font-mono); font-weight:700;">${avgPitch > 0 ? Math.round(avgPitch) + ' Hz' : 'Unvoiced'}</td>
      <td><span class="deviation-val ${devClass}">${devText}</span></td>
      <td style="font-weight: 700; color: ${penalty > 0 ? 'var(--color-clipping)' : 'var(--color-studio)'};">${penaltyText}</td>
      <td><span class="table-desc">${comment}</span></td>
      <td><button class="play-row-btn" data-start="${sec}">Jump</button></td>
    `;
    
    els.secondTableBody.appendChild(row);
  }
  
  // Register click seek listeners
  els.secondTableBody.querySelectorAll('.play-row-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const startT = parseFloat(e.target.dataset.start);
      els.audioPlayer.currentTime = startT;
      els.audioPlayer.play();
      els.playPauseBtn.textContent = '⏸';
    });
  });
}

function highlightSecondTableRow(currentTime) {
  const currentSec = Math.floor(currentTime);
  
  els.secondTableBody.querySelectorAll('tr').forEach(r => r.classList.remove('active-row'));
  
  const row = document.getElementById(`sec-row-${currentSec}`);
  if (row) {
    row.classList.add('active-row');
    row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

// ==========================================================================
// 8. Initialization Entry Point
// ==========================================================================
window.addEventListener('DOMContentLoaded', () => {
  init3D();
  bindUIEvents();
  bindAudioEvents();
});
