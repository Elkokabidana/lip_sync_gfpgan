# --- START OF FILE Debug.py (Multi-GPU Corrected) ---

import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import threading
import queue
import librosa
import sys
import time
import os
import torch
import torchvision.transforms as transforms
from collections import deque
import traceback

# +++ وارد کردن GFPGAN +++
try:
    from gfpgan import GFPGANer
    print("GFPGAN imported successfully.")
    GFPGAN_AVAILABLE = True
except ImportError:
    print("Warning: GFPGAN not found. Install it: pip install gfpgan")
    print("GFPGAN post-processing will be disabled.")
    GFPGANer = None # برای جلوگیری از خطای NameError
    GFPGAN_AVAILABLE = False

# --- تنظیمات اولیه ---
WAV2LIP_DIR = 'Wav2Lip'
WAV2LIP_ABS_PATH = os.path.abspath(WAV2LIP_DIR)
if WAV2LIP_ABS_PATH not in sys.path: sys.path.insert(0, WAV2LIP_ABS_PATH); print(f"Path '{WAV2LIP_ABS_PATH}' added.")
try: from models import Wav2Lip; print("Wav2Lip models imported successfully.")
except ImportError as e: print(f"Import Error: {e}"); sys.exit(1)
except Exception as e: print(f"Unexpected Import Error: {e}"); sys.exit(1)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
WAV2LIP_MODEL_PATH = os.path.join(WAV2LIP_DIR, 'checkpoints', 'wav2lip_gan.pth')
print(f"Wav2Lip model path set to (GAN MODEL): {WAV2LIP_MODEL_PATH}")
if not os.path.exists(WAV2LIP_MODEL_PATH): print(f"Error: Wav2Lip GAN Model file not found at '{WAV2LIP_MODEL_PATH}'!"); sys.exit(1)

# +++ تعیین دستگاه‌ها برای Multi-GPU +++
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} CUDA device(s).")
    if gpu_count >= 2:
        print("Using 2 GPUs: cuda:0 for Wav2Lip, cuda:1 for GFPGAN.")
        wav2lip_device = 'cuda:0'
        gfpgan_device = 'cuda:1'
    elif gpu_count == 1:
        print("Found 1 GPU. Using cuda:0 for both Wav2Lip and GFPGAN.")
        wav2lip_device = 'cuda:0'
        gfpgan_device = 'cuda:0'
    else: # gpu_count == 0 should not happen if torch.cuda.is_available() is True, but as safeguard
        print("Warning: No CUDA devices detected despite availability check. Using CPU.")
        wav2lip_device = 'cpu'
        gfpgan_device = 'cpu'
else:
    print("Warning: CUDA not available. Using CPU (will be very slow).")
    wav2lip_device = 'cpu'
    gfpgan_device = 'cpu'

print(f"Wav2Lip will run on: {wav2lip_device}")
print(f"GFPGAN will run on: {gfpgan_device}")

# --- پارامترهای اصلی ---
IMG_SIZE = 96; MEL_STEP_SIZE = 16; FPS = 25; WAV2LIP_BATCH_SIZE = 2
WEBCAM_WIDTH = 640; WEBCAM_HEIGHT = 480; FORMAT = pyaudio.paInt16
CHANNELS = 1; RATE = 16000; CHUNK = int(RATE / FPS); INPUT_DEVICE_INDEX = None

# --- صف و رویداد ---
audio_queue = queue.Queue(maxsize=int(FPS*1.5)); exit_event = threading.Event()

# --- MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
# Initialize face_mesh here to ensure it's globally accessible if needed later
try:
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    print("MediaPipe FaceMesh initialized.")
except Exception as e_mp:
    print(f"Error initializing MediaPipe FaceMesh: {e_mp}")
    face_mesh = None # Handle potential initialization failure
    sys.exit(1) # Exit if face detection isn't possible

# --- بافرها ---
hop_length = 200 # مقدار hop_length برای librosa melspectrogram
AUDIO_BUFFER_MAXLEN = int(FPS * 3); print(f"Audio buffer max length set to: {AUDIO_BUFFER_MAXLEN} chunks")
audio_buffer = deque(maxlen=AUDIO_BUFFER_MAXLEN)
frame_buffer = deque(maxlen=WAV2LIP_BATCH_SIZE); bbox_buffer = deque(maxlen=WAV2LIP_BATCH_SIZE)
original_frame_buffer = deque(maxlen=WAV2LIP_BATCH_SIZE)

# --- توابع کمکی ---
def face_detect_with_landmarks(images):
    if face_mesh is None: # Check if initialization failed
        return [None] * len(images)
    all_landmarks = []; image_shapes = []
    for image in images:
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB); image_rgb.flags.writeable = False
            results = face_mesh.process(image_rgb); image_rgb.flags.writeable = True
            all_landmarks.append(results.multi_face_landmarks); image_shapes.append(image.shape)
        except Exception as e: all_landmarks.append(None); image_shapes.append(None)
    bboxes = []
    for i, face_landmarks_list in enumerate(all_landmarks):
        shape = image_shapes[i]
        if not face_landmarks_list or shape is None: bboxes.append(None); continue
        face_landmarks = face_landmarks_list[0]; ih, iw, _ = shape
        try:
            lm_mouth_left = face_landmarks.landmark[61]; lm_mouth_right = face_landmarks.landmark[291]
            lm_lip_upper = face_landmarks.landmark[0]; lm_lip_lower = face_landmarks.landmark[17]
            mouth_left_x = int(lm_mouth_left.x * iw); mouth_right_x = int(lm_mouth_right.x * iw)
            lip_upper_y = int(lm_lip_upper.y * ih); lip_lower_y = int(lm_lip_lower.y * ih)
            mouth_cx = (mouth_left_x + mouth_right_x) // 2; mouth_cy = (lip_upper_y + lip_lower_y) // 2
            mouth_width = mouth_right_x - mouth_left_x
            if mouth_width <= 0: bboxes.append(None); continue
            size = int(mouth_width * 2.5); center_x = mouth_cx; center_y = mouth_cy + int(size * 0.1) # Offset center slightly down
        except IndexError: bboxes.append(None); continue # Handle cases where landmarks might be missing
        except Exception as e_lm: print(f"Landmark calculation error: {e_lm}"); bboxes.append(None); continue
        half_size = size // 2; x1 = max(0, center_x - half_size); y1 = max(0, center_y - half_size)
        x2 = min(iw, center_x + half_size); y2 = min(ih, center_y + half_size)
        if (x2 - x1) > 0 and (y2 - y1) > 0: bboxes.append([x1, y1, x2, y2])
        else: bboxes.append(None)
    return bboxes

def preprocess_frames(frames, bboxes):
    preprocessed_frames = []; valid_indices = []
    for i, (frame, bbox) in enumerate(zip(frames, bboxes)):
        if bbox is None: continue
        x1, y1, x2, y2 = map(int, bbox)
        if x1 >= x2 or y1 >= y2: continue
        face_crop = frame[y1:y2, x1:x2];
        if face_crop.size == 0: continue
        try:
             face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA);
             face_resized_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
             # Normalize [-1, 1] for GAN model
             face_tensor = torch.FloatTensor(face_resized_rgb).permute(2, 0, 1) / 255.0;
             face_normalized = (face_tensor - 0.5) * 2.0
             preprocessed_frames.append(face_normalized); valid_indices.append(i)
        except Exception as e:
             # print(f"Error preprocessing frame {i}: {e}") # Optional debug print
             continue
    if not preprocessed_frames: return None, None
    try: return torch.stack(preprocessed_frames).unsqueeze(0), valid_indices # Add batch dim: [1, B, C, H, W]
    except Exception as e: print(f"Stacking Error during preprocessing: {e}"); return None, None

def get_mel_chunk(audio_data_bytes):
    try:
        # Ensure correct dtype and normalization
        audio_signal = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    except ValueError as e:
        print(f"Audio Conversion Error: {e}, length: {len(audio_data_bytes)}"); return None

    n_fft, win_length, n_mels = 800, 800, 80; fmin, fmax = 55, 7600
    expected_signal_len = MEL_STEP_SIZE * hop_length # Should be 16 * 200 = 3200

    if len(audio_signal) < expected_signal_len:
        # print(f"Padding audio signal from {len(audio_signal)} to {expected_signal_len}") # Debug
        audio_signal = np.pad(audio_signal, (0, expected_signal_len - len(audio_signal)), 'constant', constant_values=0.0)
    elif len(audio_signal) > expected_signal_len:
        # print(f"Truncating audio signal from {len(audio_signal)} to {expected_signal_len}") # Debug
        audio_signal = audio_signal[:expected_signal_len]

    # Add check after padding/truncating
    if len(audio_signal) != expected_signal_len:
         print(f"Error: Audio signal length mismatch after adjustment! Is {len(audio_signal)}, expected {expected_signal_len}")
         return None # Critical if length is wrong before librosa

    try:
        mel = librosa.feature.melspectrogram(y=audio_signal, sr=RATE, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels, fmin=fmin, fmax=fmax, center=False)
        mel_db = librosa.power_to_db(mel, ref=np.max)
    except Exception as e:
        print(f"Librosa Error: {e}"); return None

    # Check shape *after* mel calculation
    if mel_db.shape[1] != MEL_STEP_SIZE:
        print(f"Warning: Mel shape mismatch! Expected {MEL_STEP_SIZE}, got {mel_db.shape[1]}. Adjusting...") # Keep the warning
        target_len = MEL_STEP_SIZE
        current_len = mel_db.shape[1]
        if current_len < target_len:
            mel_db = np.pad(mel_db, ((0, 0), (0, target_len - current_len)), mode='constant', constant_values=-80.0) # Pad with silence
        else:
            mel_db = mel_db[:, :target_len] # Truncate

    if not np.isfinite(mel_db).all():
        mel_db = np.nan_to_num(mel_db, nan=-80.0, posinf=-80.0, neginf=-80.0)

    # Return tensor shape [1, 1, n_mels, MEL_STEP_SIZE]
    return torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0)

def list_audio_input_devices():
    p = pyaudio.PyAudio(); numdevices=0; available_indices = []
    try: info = p.get_host_api_info_by_index(0); numdevices = info.get('deviceCount', 0)
    except Exception as e: print(f"Host API Error: {e}"); p.terminate(); return []
    print("-" * 60 + "\nAvailable Audio Input Devices:\n" + "-" * 60)
    found_input_device = False
    for i in range(0, numdevices):
        try:
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                found_input_device = True; available_indices.append(i); device_name_raw = device_info.get('name'); device_name = "Unknown"
                if isinstance(device_name_raw, bytes):
                    try: device_name = device_name_raw.decode('utf-8', errors='replace')
                    except UnicodeDecodeError: device_name = device_name_raw.decode('latin-1', errors='replace')
                elif isinstance(device_name_raw, str): device_name = device_name_raw
                print(f"  Index {i}: {device_name}")
        except Exception as e: print(f"  Dev Info Err {i}: {e}")
    if not found_input_device: print("No active input devices found.")
    print("-" * 60); p.terminate(); return available_indices

def record_audio(device_index, stop_event):
    p = pyaudio.PyAudio(); stream = None
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=device_index)
        print(f"🎤 Recording started from Index {device_index}...")
    except Exception as e:
        print(f"Stream Open Err: {e}"); p.terminate(); stop_event.set(); return

    while not stop_event.is_set():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            if stop_event.is_set(): break
            try: audio_queue.put(data, block=False)
            except queue.Full:
                 try:
                     audio_queue.get_nowait() # Discard oldest
                     audio_queue.put_nowait(data) # Add newest
                 except queue.Empty: pass
                 except Exception as qe: print(f"Queue error after full: {qe}")
        except IOError as e:
            if hasattr(e, 'errno') and e.errno == pyaudio.paInputOverflowed:
                 time.sleep(0.001)
            elif hasattr(e, 'errno') and e.errno == pyaudio.paStreamIsStopped: print("Audio stream stopped."); stop_event.set(); break
            else: print(f"IO Err Aud Rec: {e}"); time.sleep(0.01)
        except Exception as e: print(f"Unknown Err Aud Rec: {e}"); stop_event.set(); break

    print(f"🎤 Recording stopped for Index {device_index}.")
    if stream:
        try:
            if stream.is_active(): stream.stop_stream()
            stream.close()
        except Exception as e: print(f"Error closing audio stream: {e}")
    try: p.terminate()
    except Exception as e: print(f"Error terminating PyAudio: {e}")
    print("Audio recording thread finished.")

# +++ بارگذاری مدل GFPGAN روی دستگاه مشخص +++
face_enhancer = None
if GFPGAN_AVAILABLE:
    print("="*30 + " GFPGAN Model Loading " + "="*30)
    try:
        # مسیر فایل محلی - نام فایل را با نام فایل واقعی خود مطابقت دهید
        gfpgan_local_filename = 'GFPGANv1.3.pth' # یا 'GFPGANV1.3.pth' اگر با V بزرگ است
        gfpgan_folder_name = 'gfpgan'
        gfpgan_local_path = os.path.join(gfpgan_folder_name, gfpgan_local_filename)
        print(f"Attempting to load GFPGAN from local path: {gfpgan_local_path}")

        gfpgan_upscale = 1
        face_enhancer = GFPGANer(
            model_path=gfpgan_local_path,
            upscale=gfpgan_upscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            # استفاده از دستگاه مشخص شده برای GFPGAN
            device=gfpgan_device
        )
        print(f"GFPGAN model loaded successfully from '{gfpgan_local_path}' onto {gfpgan_device}.")
    except FileNotFoundError:
         print(f"!!!!!!!! ERROR: GFPGAN model file not found at '{gfpgan_local_path}'. Check path/filename. !!!!!!!!!")
         print("GFPGAN post-processing will be disabled.")
         face_enhancer = None
    except Exception as e:
        # چاپ traceback برای دیدن جزئیات بیشتر خطا در صورت بروز خطای غیرمنتظره
        print(f"!!!!!!!! Error loading GFPGAN model from local file: {e} !!!!!!!!!")
        traceback.print_exc() # <-- اضافه کردن traceback
        print("GFPGAN post-processing will be disabled.")
        face_enhancer = None
    # این print باید بعد از بلوک try-except باشد
    print("="*70) # این خط حالا باید بدون خطا اجرا شود

# --- تابع اصلی پردازش ویدیو ---
def process_video(stop_event, model):
    # تعریف متغیرهای global لازم در ابتدای تابع
    global face_enhancer, wav2lip_device, gfpgan_device
    global audio_buffer, frame_buffer, bbox_buffer, original_frame_buffer

    cap = cv2.VideoCapture(0);
    if not cap.isOpened(): print("Error: Cannot open webcam."); stop_event.set(); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT); cap.set(cv2.CAP_PROP_FPS, FPS)
    actual_fps = cap.get(cv2.CAP_PROP_FPS); effective_fps = actual_fps if actual_fps > 0 else FPS; print(f"Requested FPS: {FPS}, Actual FPS: {effective_fps}")

    MASK_START_RATIO = 0.65
    MASK_BLUR_KERNEL_SIZE = (31, 31)
    mask_blurred_precalculated = None
    try:
        base_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32); mask_start_row = int(IMG_SIZE * MASK_START_RATIO); base_mask[mask_start_row:, :] = 1.0;
        if (MASK_BLUR_KERNEL_SIZE[0] > 0 and MASK_BLUR_KERNEL_SIZE[1] > 0 and MASK_BLUR_KERNEL_SIZE[0]%2!=0 and MASK_BLUR_KERNEL_SIZE[1]%2!=0):
            mask_blurred_precalculated = cv2.GaussianBlur(base_mask, MASK_BLUR_KERNEL_SIZE, 0); print("Pre-calculated blurred mask generated.")
        else: print("Warning: Invalid mask blur kernel size. Blur disabled."); mask_blurred_precalculated = base_mask
    except Exception as e:
        print(f"Mask Precomputation Error: {e}");
        mask_blurred_precalculated = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    print("📸 Video processing and Lip Sync (Wav2Lip+GAN+Blend+GFPGAN) started...")
    last_known_good_bbox = None; generated_face_cache = {}
    fps_counter = 0; start_time = time.time(); display_fps = 0.0

    while cap.isOpened() and not stop_event.is_set():
        audio_read_count = 0;
        while not audio_queue.empty() and audio_read_count < WAV2LIP_BATCH_SIZE * 2 :
            try: audio_buffer.append(audio_queue.get_nowait()); audio_read_count += 1
            except queue.Empty: break

        ret, frame = cap.read();
        if not ret: time.sleep(0.01); continue
        frame = cv2.flip(frame, 1)
        current_bbox_list = face_detect_with_landmarks([frame])
        current_bbox = current_bbox_list[0] if current_bbox_list else None

        if current_bbox is not None: last_known_good_bbox = current_bbox
        else: current_bbox = last_known_good_bbox

        if current_bbox is None:
            cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Real-time Lip Sync (GAN+Blend+GFPGAN)', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): print("'q' pressed. Exiting..."); stop_event.set(); break
            continue

        frame_buffer.append(frame.copy()); original_frame_buffer.append(frame.copy()); bbox_buffer.append(current_bbox)

        output_frame = None
        if len(frame_buffer) == WAV2LIP_BATCH_SIZE:
            frames_to_process = list(frame_buffer); bboxes_to_process = list(bbox_buffer)
            original_frames_batch = list(original_frame_buffer)

            face_batch_cpu, valid_indices = preprocess_frames(frames_to_process, bboxes_to_process)

            mel_chunk = None
            bytes_per_sample = pyaudio.get_sample_size(FORMAT)
            num_samples_needed = MEL_STEP_SIZE * hop_length
            num_bytes_needed = num_samples_needed * bytes_per_sample * CHANNELS
            bytes_per_chunk = CHUNK * bytes_per_sample * CHANNELS
            num_chunks_to_take = int(np.ceil(num_bytes_needed / bytes_per_chunk)) if bytes_per_chunk > 0 else 1
            num_chunks_in_buffer = len(audio_buffer)

            if num_chunks_in_buffer >= num_chunks_to_take:
                audio_segment_chunks = list(audio_buffer)[-num_chunks_to_take:]
                audio_segment_bytes = b''.join(audio_segment_chunks)
                if len(audio_segment_bytes) >= num_bytes_needed:
                    input_bytes_for_mel = audio_segment_bytes[-num_bytes_needed:]
                    mel_chunk_cpu = get_mel_chunk(input_bytes_for_mel)
                    if mel_chunk_cpu is not None:
                        mel_chunk = mel_chunk_cpu.to(wav2lip_device) # انتقال به GPU Wav2Lip
                # else: print("Warn: Not enough bytes for mel") # Debug
            # else: print("Warn: Not enough audio chunks") # Debug

            generated_faces = None; generated_face_cache.clear()
            if face_batch_cpu is not None and mel_chunk is not None and valid_indices:
                face_batch = face_batch_cpu.to(wav2lip_device) # انتقال بچ چهره به GPU Wav2Lip
                with torch.no_grad():
                    try:
                        generated_faces_batch = model(mel_chunk, face_batch) # اجرا روی wav2lip_device
                        # انتقال خروجی به CPU برای پردازش OpenCV
                        generated_faces = generated_faces_batch.squeeze(0).cpu().numpy();
                        generated_faces = np.transpose(generated_faces, (0, 2, 3, 1))
                        generated_faces = np.clip((generated_faces + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
                        for i, face_idx in enumerate(valid_indices):
                            if i < len(generated_faces): generated_face_cache[face_idx] = generated_faces[i]
                    except Exception as e:
                        print(f"!!!!!!!!! Wav2Lip Model Exec Error: {e} !!!!!!!!!"); traceback.print_exc()

            output_frame = original_frames_batch[0].copy();
            output_bbox = bboxes_to_process[0];
            output_face_to_paste_idx_0 = generated_face_cache.get(0)

            # نمایش خروجی خام مدل (اختیاری)
            # if output_face_to_paste_idx_0 is not None:
            #      try:
            #          raw_face_display = cv2.cvtColor(output_face_to_paste_idx_0, cv2.COLOR_RGB2BGR)
            #          cv2.imshow("Raw Model Output (96x96)", raw_face_display)
            #      except Exception as e_raw: pass

            # Alpha Blending (روی CPU)
            if output_face_to_paste_idx_0 is not None and output_bbox is not None and mask_blurred_precalculated is not None:
                x1, y1, x2, y2 = map(int, output_bbox);
                x1, y1 = max(x1, 0), max(y1, 0);
                x2, y2 = min(x2, output_frame.shape[1]), min(y2, output_frame.shape[0])
                if x1 < x2 and y1 < y2:
                    target_h, target_w = y2 - y1, x2 - x1
                    try:
                        gen_face_resized = cv2.resize(output_face_to_paste_idx_0, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                        try: gen_face_resized_bgr = cv2.cvtColor(gen_face_resized, cv2.COLOR_RGB2BGR)
                        except cv2.error: gen_face_resized_bgr = gen_face_resized
                        mask_resized = cv2.resize(mask_blurred_precalculated, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        mask_resized = mask_resized[:, :, np.newaxis]
                        original_face_roi = output_frame[y1:y2, x1:x2]
                        blended_face = np.clip(
                            original_face_roi.astype(np.float32) * (1.0 - mask_resized) +
                            gen_face_resized_bgr.astype(np.float32) * mask_resized,
                            0, 255
                        ).astype(np.uint8)
                        output_frame[y1:y2, x1:x2] = blended_face
                    except Exception as e: print(f"Alpha Blending Error: {e}")
            elif output_bbox is not None:
                 x1_text, y1_text, _, _ = map(int, output_bbox)
                 cv2.putText(output_frame, "Sync?", (max(0,x1_text), max(0, y1_text-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            frame_buffer.popleft(); bbox_buffer.popleft(); original_frame_buffer.popleft()

        else: # Buffering...
            output_frame = frame.copy()
            cv2.putText(output_frame, f"Buffering... {len(frame_buffer)}/{WAV2LIP_BATCH_SIZE}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # پس‌پردازش با GFPGAN (روی gfpgan_device)
        display_frame_final = output_frame
        if face_enhancer is not None and output_frame is not None:
            try:
                _, _, restored_img = face_enhancer.enhance(
                    output_frame, has_aligned=False, only_center_face=False, paste_back=True
                )
                if restored_img is not None: display_frame_final = restored_img
            except Exception as e_gfpgan:
                # print(f"GFPGAN Enhance Error: {e_gfpgan}") # Debug Optional
                display_frame_final = output_frame # Fallback

        # نمایش فریم نهایی
        if display_frame_final is not None:
            fps_counter += 1
            if (time.time() - start_time) > 1.0:
                display_fps = fps_counter / (time.time() - start_time)
                fps_counter = 0
                start_time = time.time()
            cv2.putText(display_frame_final, f"FPS: {display_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Real-time Lip Sync (GAN+Blend+GFPGAN)', display_frame_final)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): print("'q' pressed. Exiting..."); stop_event.set(); break

    # --- تمیزکاری ---
    print("Releasing webcam and destroying windows...")
    cap.release();
    cv2.destroyAllWindows();
    if face_mesh is not None and hasattr(face_mesh, 'close'):
        try: face_mesh.close(); print("MediaPipe FaceMesh closed.")
        except Exception as e: print(f"Error closing face mesh: {e}")

    # پاکسازی حافظه GPU ها
    if torch.cuda.is_available():
        try:
            # دستگاه‌هایی که واقعا استفاده شدند
            used_devices = set([d for d in [wav2lip_device, gfpgan_device] if 'cuda' in d])
            for device_id_str in used_devices:
                 with torch.cuda.device(device_id_str): torch.cuda.empty_cache()
            if used_devices: print("GPU memory cache potentially cleared for used devices:", list(used_devices))
        except Exception as e_gpu_clean: print(f"Error clearing GPU cache: {e_gpu_clean}")
    print("Video processing thread finished.")


# --- اجرای اصلی برنامه ---
if __name__ == "__main__":
    # 1. انتخاب دستگاه صوتی
    print("="*30 + " Audio Device Selection " + "="*30);
    available_indices = list_audio_input_devices(); INPUT_DEVICE_INDEX = None
    if not available_indices: print("Error: No input devices found."); sys.exit(1)
    if len(available_indices) == 1: INPUT_DEVICE_INDEX = available_indices[0]; print(f"Auto-selected Index: {INPUT_DEVICE_INDEX}")
    else:
        while INPUT_DEVICE_INDEX is None:
             try: selected = input(f"Enter microphone Index from {available_indices}: "); candidate_index = int(selected.strip())
             except ValueError: print("Invalid input. Please enter a number."); continue
             except (EOFError, KeyboardInterrupt): print("\nSelection cancelled by user."); sys.exit(0)
             if candidate_index in available_indices: INPUT_DEVICE_INDEX = candidate_index
             else: print(f"Invalid Index. Please choose from {available_indices}.")
    print(f"--> Selected Audio Device Index: {INPUT_DEVICE_INDEX}")
    print("="*70)

    # 2. بارگذاری مدل Wav2Lip روی دستگاه مشخص
    print("="*30 + " Wav2Lip Model Loading " + "="*30); print(f"Loading Wav2Lip GAN model from: {WAV2LIP_MODEL_PATH}...")
    wav2lip_model = None
    try:
        wav2lip_model = Wav2Lip(); checkpoint = torch.load(WAV2LIP_MODEL_PATH, map_location=wav2lip_device) # Load directly to target device if possible
        if "state_dict" in checkpoint: s = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict): s = checkpoint
        else: print("Error: Unknown Wav2Lip checkpoint format."); sys.exit(1)
        new_s = {};
        for k, v in s.items(): new_s[k.replace('module.', '', 1)] = v
        wav2lip_model.load_state_dict(new_s); print("Wav2Lip state_dict loaded successfully.")
        # Ensure model is on the correct device after loading state_dict
        wav2lip_model = wav2lip_model.to(wav2lip_device);
        wav2lip_model.eval(); print(f"Wav2Lip model is on '{wav2lip_device}' and in eval mode.")
    except FileNotFoundError: print(f"Error: Wav2Lip model file not found at '{WAV2LIP_MODEL_PATH}'."); sys.exit(1)
    except Exception as e: print(f"Wav2Lip Model Loading Error: {e}"); traceback.print_exc(); sys.exit(1)
    print("="*70)

    # (مدل GFPGAN قبلاً بالاتر با دستگاه خودش بارگذاری شده است)

    # 3. شروع تردها
    print("="*30 + " Starting Threads " + "="*30); print("Starting audio recording thread...");
    audio_thread = threading.Thread(target=record_audio, args=(INPUT_DEVICE_INDEX, exit_event), daemon=True)
    audio_thread.start(); time.sleep(1)
    if not audio_thread.is_alive(): print("Error: Audio thread failed to start."); sys.exit(1)
    print("Audio thread started.")
    print("="*70)

    # 4. شروع پردازش ویدیو و حلقه اصلی
    print("="*30 + " Starting Processing " + "="*30); video_processing_successful = True
    try:
        # Warm-up Wav2Lip
        if 'cuda' in wav2lip_device: # Only warm-up if on GPU
            print(f"Performing Model Warm-up (Wav2Lip on {wav2lip_device})...");
            try:
                dummy_mel = torch.randn(1, 1, 80, MEL_STEP_SIZE, device=wav2lip_device)
                dummy_face_batch = torch.randn(1, WAV2LIP_BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=wav2lip_device)
                with torch.no_grad(): _ = wav2lip_model(dummy_mel, dummy_face_batch)
                print("Wav2Lip Warm-up complete.")
            except Exception as warmup_e: print(f"Warning: Wav2Lip Warm-up failed: {warmup_e}. Continuing...")

        # Warm-up GFPGAN
        if face_enhancer is not None and 'cuda' in gfpgan_device: # Only warm-up if on GPU
            print(f"Performing Model Warm-up (GFPGAN on {gfpgan_device})...");
            try:
                dummy_frame = np.zeros((48, 48, 3), dtype=np.uint8) # Smaller dummy frame for faster warmup
                with torch.no_grad(): _ = face_enhancer.enhance(dummy_frame, paste_back=True)
                print("GFPGAN Warm-up complete.")
            except Exception as warmup_gfpgan_e: print(f"Warning: GFPGAN Warm-up failed: {warmup_gfpgan_e}. Continuing...")

        print("="*70);
        process_video(exit_event, wav2lip_model)

    except KeyboardInterrupt: print("\nUser interrupt (Ctrl+C)."); video_processing_successful = False
    except Exception as e: print(f"\nCritical Error in main loop: {e}"); traceback.print_exc(); video_processing_successful = False
    finally:
        print("\n" + "="*30 + " Cleaning Up and Exiting " + "="*30);
        exit_event.set()

        if 'audio_thread' in locals() and audio_thread.is_alive():
            print("Waiting for audio thread to finish...");
            audio_thread.join(timeout=2.0);
            if audio_thread.is_alive(): print("Warning: Audio thread did not terminate gracefully.")

        # پاک کردن مدل‌ها و حافظه GPU
        try:
            del wav2lip_model
            del face_enhancer # اطمینان از حذف رفرنس
            if torch.cuda.is_available():
                used_devices = set([d for d in [wav2lip_device, gfpgan_device] if 'cuda' in d])
                for device_id_str in used_devices:
                     with torch.cuda.device(device_id_str): torch.cuda.empty_cache()
                if used_devices: print("Final GPU memory cache potentially cleared.")
        except Exception as e_final_clean: print(f"Error during final cleanup: {e_final_clean}")

        status_message = "Successfully" if video_processing_successful else "with Errors/Interruption"
        print(f"Program finished {status_message}.")
        print("="*70)

# --- END OF FILE Debug.py (Multi-GPU Corrected) ---