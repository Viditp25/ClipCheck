import os
import sys
import json
import numpy as np
import librosa

class ClipCheckEngine:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
        # Audio QA Thresholds
        self.clipping_threshold_db = -0.1
        self.silence_threshold_db = -50.0
        
        # Jitter/Glitch Thresholds
        self.robotic_jitter_threshold = 0.45
        self.spectral_discontinuity_threshold = 2.5
        
    def analyze(self, audio_path):
        """Runs the complete ClipCheck DSP analysis pipeline on the target file."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        print(f"Ingesting file: {audio_path}")
        # Ingestion & Standardization (Step 1 & 2)
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        
        print(f"File loaded. Duration: {duration:.2s}s, Sample Rate: {sr}Hz")
        print("Extracting acoustic features...")
        
        # Frame settings (100ms frames)
        frame_len = int(sr * 0.1) # 2205 samples
        hop_len = frame_len
        total_frames = len(y) // frame_len
        
        # Extract features (Step 3)
        # 3A - MFCC Extraction
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_len*2, hop_length=hop_len)
        
        # 3C - RMS Energy
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        
        # 3D - Spectral Centroid & Bandwidth
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_len*2, hop_length=hop_len)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=frame_len*2, hop_length=hop_len)[0]
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=frame_len*2, hop_length=hop_len)[0]
        
        # 3B - Fundamental Frequency (F0) Pitch Tracking
        # YIN is robust and fast for fundamental frequency tracking in localized voice tracks
        try:
            f0 = librosa.yin(y=y, fmin=60, fmax=400, sr=sr, frame_length=frame_len*2, hop_length=hop_len)
            # YIN outputs nan or fmin if unvoiced, clean up values below fmin
            f0[np.isnan(f0)] = 0
        except Exception as e:
            # Fallback if YIN fails due to silence
            f0 = np.zeros(total_frames)

        # Run Detectors (Step 6)
        clipping_blocks = self._detect_clipping(y, frame_len, total_frames)
        silence_blocks = self._detect_silence(rms)
        glitch_blocks = self._detect_robotic_glitches(mfcc, centroid, flatness, f0, rms)
        
        # Tone & Emotion Classification
        emotions, avg_emotion = self._analyze_emotions(f0, rms, y, sr)
        
        # Compile Timestamp log (Step 8)
        errors = self._generate_timestamp_log(
            clipping_blocks, silence_blocks, glitch_blocks, emotions, duration, total_frames
        )
        
        # Compute final Quality Score (Step 7)
        score = self._compute_quality_score(
            clipping_blocks, silence_blocks, glitch_blocks, errors, total_frames
        )
        
        # Summary calculations
        peak_amp = np.max(np.abs(y))
        peak_db = 20 * np.log10(peak_amp) if peak_amp > 0 else -100
        avg_rms = np.mean(rms)
        avg_db = 20 * np.log10(avg_rms) if avg_rms > 0 else -100
        
        report = {
            "file_name": os.path.basename(audio_path),
            "duration_seconds": round(duration, 2),
            "quality_score": score,
            "peak_dbfs": round(float(peak_db), 1),
            "average_dbfs": round(float(avg_db), 1),
            "clipping_blocks_count": int(np.sum(clipping_blocks)),
            "dead_air_seconds": round(float(np.sum(silence_blocks) * 0.1), 2),
            "overall_vibe": avg_emotion,
            "anomalies": errors
        }
        
        return report

    def _detect_clipping(self, y, frame_len, total_frames):
        """Identifies frames containing clipping (samples close to 1.0 or -1.0)."""
        clipping = np.zeros(total_frames, dtype=bool)
        for i in range(total_frames):
            frame_y = y[i * frame_len : (i + 1) * frame_len]
            if len(frame_y) > 0:
                peak = np.max(np.abs(frame_y))
                peak_db = 20 * np.log10(peak) if peak > 0 else -100
                if peak_db >= self.clipping_threshold_db:
                    clipping[i] = True
        return clipping

    def _detect_silence(self, rms):
        """Identifies silent/dead-air frames using RMS thresholding."""
        rms_db = 20 * np.log10(rms + 1e-6)
        return rms_db < self.silence_threshold_db

    def _detect_robotic_glitches(self, mfcc, centroid, flatness, f0, rms):
        """Detects robotic AI voice synthesis glitches and compression artifacts."""
        total_frames = len(rms)
        glitches = np.zeros(total_frames, dtype=bool)
        
        # Standardize features for variance checks
        mfcc_diff = np.diff(mfcc, axis=1)
        # Compute MFCC variance over sliding window (5 frames = 500ms)
        win_size = 5
        
        for i in range(win_size, total_frames - win_size):
            # Skip silent frames (only analyze active speech)
            rms_db = 20 * np.log10(rms[i] + 1e-6)
            if rms_db < -35:
                continue
                
            # Robotic glitches exhibit unnaturally static speech textures (low MFCC variance)
            # or extreme high-frequency metallic resonance (high spectral flatness + high centroid)
            local_mfcc_var = np.var(mfcc[:, i - win_size : i + win_size])
            local_centroid = centroid[i]
            local_flatness = flatness[i]
            
            # Autocorrelation pitch changes
            local_pitch_diff = abs(f0[i] - f0[i-1]) if f0[i] > 0 and f0[i-1] > 0 else 0
            
            # Condition 1: Extremely flat/static timbre (robot voice signature)
            is_robotic_timbre = local_mfcc_var < 5.0 and f0[i] > 0
            
            # Condition 2: High spectral flat metallic noise + high centroid
            is_metallic_artifact = local_flatness > 0.08 and local_centroid > 3500
            
            # Condition 3: Unnatural pitch transition (digital jitter jump)
            is_pitch_jitter = local_pitch_diff > 120 # sudden jump greater than 120Hz in 100ms
            
            if is_robotic_timbre or is_metallic_artifact or is_pitch_jitter:
                glitches[i] = True
                
        return glitches

    def _analyze_emotions(self, f0, rms, y, sr):
        """Classifies localized blocks into emotional tones: Cheer, Serious, Neutral, Sad, Angry."""
        total_frames = len(rms)
        emotions = []
        
        # Calculate pitch baseline
        voiced_pitches = f0[f0 > 0]
        mean_pitch = np.mean(voiced_pitches) if len(voiced_pitches) > 0 else 120.0
        pitch_std = np.std(voiced_pitches) if len(voiced_pitches) > 0 else 10.0
        
        # Classify each frame
        for i in range(total_frames):
            p = f0[i]
            r = rms[i]
            r_db = 20 * np.log10(r + 1e-6)
            
            if r_db < -45:
                emotions.append("Silent")
                continue
                
            if p == 0:
                emotions.append("Neutral")
                continue
                
            # Decision Tree based on pitch dynamics & energy
            if p > (mean_pitch + pitch_std) and r_db > -18:
                emotions.append("Angry/Excited")
            elif p > mean_pitch and r_db > -25:
                emotions.append("Cheerful")
            elif p < (mean_pitch - 0.5 * pitch_std) and r_db < -30:
                emotions.append("Sad")
            elif p >= (mean_pitch - 0.5 * pitch_std) and p <= mean_pitch and r_db >= -28:
                emotions.append("Serious")
            else:
                emotions.append("Neutral")
                
        # Determine overall dominant emotion
        valid_emotions = [e for e in emotions if e not in ["Silent", "Neutral"]]
        if not valid_emotions:
            overall = "Neutral"
        else:
            values, counts = np.unique(valid_emotions, return_counts=True)
            overall = values[np.argmax(counts)]
            
        return emotions, overall

    def _generate_timestamp_log(self, clipping, silence, glitches, emotions, duration, total_frames):
        """Groups sequential frame alerts into clean, timestamped anomaly logs."""
        errors = []
        block_dur = 0.1
        
        # Helper: Merge adjacent frames of specific error flags
        def extract_intervals(flag_arr, error_type, severity, desc_fn):
            intervals = []
            start_idx = None
            
            for idx in range(total_frames):
                if flag_arr[idx]:
                    if start_idx is None:
                        start_idx = idx
                else:
                    if start_idx is not None:
                        intervals.append({
                            "start": round(start_idx * block_dur, 2),
                            "end": round(idx * block_dur, 2),
                            "type": error_type,
                            "severity": severity,
                            "description": desc_fn(start_idx, idx)
                        })
                        start_idx = None
                        
            # Close pending
            if start_idx is not None:
                intervals.append({
                    "start": round(start_idx * block_dur, 2),
                    "end": round(total_frames * block_dur, 2),
                    "type": error_type,
                    "severity": severity,
                    "description": desc_fn(start_idx, total_frames)
                })
            return intervals

        # 1. Clipping Logs
        errors.extend(extract_intervals(
            clipping, "Clipping Distortion", "High",
            lambda s, e: f"Volume Peaked above {self.clipping_threshold_db}dBFS causing digital ceiling square distortion."
        ))
        
        # 2. Silence Logs
        errors.extend(extract_intervals(
            silence, "Dead Air", "Medium",
            lambda s, e: f"Continuous silence zone ({round((e-s)*block_dur, 2)}s) with absolute drop-out below {self.silence_threshold_db}dBFS."
        ))
        
        # 3. Robotic Voice Logs
        errors.extend(extract_intervals(
            glitches, "Robotic AI Jitter", "High",
            lambda s, e: "Metallic digital resonance, static speech timbre, or unnatural synthetic frequency shifts detected."
        ))
        
        # Sort errors chronologically by start timestamp
        errors.sort(key=lambda x: x["start"])
        return errors

    def _compute_quality_score(self, clipping, silence, glitches, errors, total_frames):
        """Calculates final QA score (0-100) based on severity weighting of anomalies."""
        score = 100
        
        # Accumulate totals
        clip_count = np.sum(clipping)
        glitch_count = np.sum(glitches)
        
        # Count dead air duration
        silence_frames = np.sum(silence)
        silence_seconds = silence_frames * 0.1
        
        # Proportional deductions
        total_duration = total_frames * 0.1
        
        clipping_deduction = (clip_count / total_frames) * 400.0 if total_frames > 0 else 0.0
        glitch_deduction = (glitch_count / total_frames) * 100.0 if total_frames > 0 else 0.0
        dead_air_deduction = (silence_seconds / total_duration) * 300.0 if total_duration > 0 else 0.0
        
        deductions = clipping_deduction + glitch_deduction + dead_air_deduction
        score = int(round(max(0, min(100, score - deductions))))
        return score

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python engine.py <path_to_audio_file> [output_report.json]")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "clipcheck_report.json"
    
    engine = ClipCheckEngine()
    try:
        report = engine.analyze(audio_file)
        
        # Save output JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4)
            
        print("\n=== CLIPCHECK QA REPORT ===")
        print(f"File Name:     {report['file_name']}")
        print(f"Duration:      {report['duration_seconds']} seconds")
        print(f"Overall Vibe:  {report['overall_vibe']}")
        print(f"Quality Score: {report['quality_score']}/100")
        print(f"Peak Level:    {report['peak_dbfs']} dBFS")
        print(f"Average Level: {report['average_dbfs']} dBFS")
        print(f"Clipping:      {report['clipping_blocks_count']} blocks")
        print(f"Dead Air:      {report['dead_air_seconds']} seconds")
        print(f"Anomalies:     {len(report['anomalies'])} logged errors")
        print(f"Full report written to: {output_file}\n")
        
    except Exception as e:
        print(f"Error running ClipCheck engine: {e}")
        sys.exit(1)
