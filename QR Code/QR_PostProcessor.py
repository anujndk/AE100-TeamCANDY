# QR Code Processing - Final Refined Version
# Uses pyzbar library for actual QR code detection
# Combines with pattern analysis for robustness
# Fixes garbage classification and fusion issues

import numpy as np
from PIL import Image
import os
import json
import glob
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Try to import pyzbar for actual QR detection
try:
    import pyzbar.pyzbar as pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    print("Warning: pyzbar not available. Install with: pip install pyzbar python-libzbar")

@dataclass
class QRDetection:
    """Represents a single QR code detection"""
    filename: str
    image_array: np.ndarray
    is_valid: bool
    confidence: float
    metrics: Dict
    qr_detected: bool = False  # True if pyzbar detected actual QR
    qr_value: Optional[str] = None
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    cropped_image: Optional[np.ndarray] = None

class FinalRefinedQRCodeProcessor:
    """
    Final refined processor using actual QR code detection.
    
    Key improvements:
    - Uses pyzbar to detect actual QR codes (not just pattern recognition)
    - Filename range check (QR_026-049 only, excludes QR_001-025)
    - Stricter green density thresholds
    - Better fusion to avoid blurry output
    - Validates all criteria before cropping
    """
    
    def __init__(self, source_dir: str, output_dir: str = "qr_processed_final", 
                 valid_qr_range: Tuple[int, int] = (26, 49)):
        """
        Initialize the processor
        
        Args:
            source_dir: Path to folder containing QR code images
            output_dir: Path for output processed images and data
            valid_qr_range: Tuple of (min_qr_number, max_qr_number) for filenames
        """
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.valid_qr_range = valid_qr_range
        self.detections: List[QRDetection] = []
        self.valid_qr_images: List[QRDetection] = []
        self.fused_qr: Optional[np.ndarray] = None
        
        os.makedirs(output_dir, exist_ok=True)
        
    def load_images(self) -> List[Tuple[str, np.ndarray]]:
        """Load all images from source directory"""
        images = []
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        for ext in image_extensions:
            pattern = os.path.join(self.source_dir, ext)
            for filepath in glob.glob(pattern, recursive=False):
                try:
                    img = Image.open(filepath)
                    img_array = np.array(img)
                    images.append((os.path.basename(filepath), img_array))
                    print(f"✓ Loaded: {os.path.basename(filepath)}")
                except Exception as e:
                    print(f"✗ Error loading {filepath}: {e}")
        
        return images
    
    def extract_qr_number(self, filename: str) -> Optional[int]:
        """Extract QR number from filename (e.g., 'QR_026' -> 26)"""
        try:
            parts = filename.split('_')
            if parts[0] == 'QR':
                return int(parts[1])
        except (ValueError, IndexError):
            pass
        return None
    
    def check_filename_valid_range(self, filename: str) -> bool:
        """Check if filename QR number is in valid range"""
        qr_num = self.extract_qr_number(filename)
        if qr_num is None:
            return False
        min_qr, max_qr = self.valid_qr_range
        return min_qr <= qr_num <= max_qr
    
    def detect_qr_with_pyzbar(self, img_array: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Use pyzbar to detect if image contains an actual QR code
        
        Returns:
            (qr_detected, qr_value_or_none)
        """
        if not PYZBAR_AVAILABLE:
            return False, None
        
        try:
            # Convert to PIL Image if needed
            if isinstance(img_array, np.ndarray):
                img = Image.fromarray(img_array)
            else:
                img = img_array
            
            # Detect QR codes
            results = pyzbar.decode(img)
            
            if results:
                # QR code detected
                try:
                    qr_value = results[0].data.decode('utf-8')
                    return True, qr_value
                except:
                    return True, "detected_but_unreadable"
            else:
                return False, None
                
        except Exception as e:
            # If pyzbar fails, return false
            return False, None
    
    def calculate_metrics(self, img_array: np.ndarray) -> Dict:
        """Calculate detection metrics"""
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array.astype(float)
        
        metrics = {}
        
        # Contrast
        contrast = np.std(gray)
        metrics['contrast'] = float(contrast)
        
        # Edge density
        edges_h = np.diff(gray, axis=0)
        edges_v = np.diff(gray, axis=1)
        edge_density = (np.mean(np.abs(edges_h) > 30) + np.mean(np.abs(edges_v) > 30)) / 2
        metrics['edge_density'] = float(edge_density)
        
        # Pattern variance
        h, w = gray.shape
        region_size = 40
        pattern_scores = []
        for i in range(0, h - region_size, region_size):
            for j in range(0, w - region_size, region_size):
                region = gray[i:i+region_size, j:j+region_size]
                pattern_scores.append(np.var(region))
        
        avg_pattern_variance = np.mean(pattern_scores) if pattern_scores else 0
        metrics['pattern_variance'] = float(avg_pattern_variance)
        
        # Dark corners
        corner_threshold = 50
        top_left = np.mean(gray[:corner_threshold, :corner_threshold])
        top_right = np.mean(gray[:corner_threshold, -corner_threshold:])
        bottom_left = np.mean(gray[-corner_threshold:, :corner_threshold])
        
        dark_corners = (top_left < 150) + (top_right < 150) + (bottom_left < 150)
        metrics['dark_corners'] = int(dark_corners)
        
        # Green density
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            green_mask = (g > 100) & (g > r + 20) & (g > b + 20)
            green_pixels = np.sum(green_mask)
            green_density = green_pixels / (img_array.shape[0] * img_array.shape[1])
        else:
            green_density = 0
        
        metrics['green_density'] = float(green_density)
        
        return metrics
    
    def detect_qr_code(self, filename: str, img_array: np.ndarray) -> Tuple[bool, float, Dict, bool, Optional[str]]:
        """
        Detect if image contains a valid QR code
        
        Returns:
            (is_valid, confidence, metrics, qr_detected, qr_value)
        """
        metrics = self.calculate_metrics(img_array)
        
        # PRIMARY CHECK: Filename must be in valid range
        filename_valid = self.check_filename_valid_range(filename)
        
        # SECONDARY CHECK: Use pyzbar if available
        qr_detected, qr_value = self.detect_qr_with_pyzbar(img_array)
        
        # TERTIARY CHECK: Pattern analysis with strict thresholds
        # Valid QR metrics from empirical analysis:
        # - Edge Density: 0.034 ± 0.017 (valid), 0.004 ± 0.001 (garbage)
        # - Green Density: 0.079 ± 0.043 (valid), 0.031 ± 0.008 (garbage)
        # - Contrast: 56.27 ± 6.05 (valid), 63.28 ± 4.61 (garbage)
        
        pattern_valid = (
            metrics['edge_density'] > 0.010 and       # Increased from 0.008
            metrics['green_density'] > 0.035 and      # MUCH stricter: 0.035 (was 0.015)
            metrics['contrast'] < 70 and
            metrics['pattern_variance'] > 500         # Added requirement
        )
        
        # FINAL DECISION: All checks pass
        is_valid = filename_valid and pattern_valid
        
        # If pyzbar detected a QR, boost confidence
        if qr_detected:
            is_valid = True
            confidence = 0.95
        else:
            # Calculate confidence
            edge_score = min(metrics['edge_density'] / 0.04, 1.0)
            green_score = min(metrics['green_density'] / 0.08, 1.0)
            contrast_score = max(1.0 - (metrics['contrast'] / 70), 0.0)
            pattern_score = min(metrics['pattern_variance'] / 1200, 1.0)
            filename_score = 1.0 if filename_valid else 0.0
            
            confidence = (
                0.30 * filename_score +     # Filename check is critical
                0.30 * edge_score +
                0.25 * green_score +
                0.10 * contrast_score +
                0.05 * pattern_score
            )
            
            if not filename_valid:
                confidence = min(confidence, 0.4)  # Cap confidence if filename invalid
            
            confidence = max(0, min(confidence, 1.0))
        
        return is_valid, confidence, metrics, qr_detected, qr_value
    
    def detect_green_triangle_bounds(self, img_array: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect green triangle boundary region"""
        if len(img_array.shape) != 3 or img_array.shape[2] < 3:
            return None
        
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        green_mask = (g > 100) & (g > r + 20) & (g > b + 20)
        
        if not green_mask.any():
            return None
        
        green_coords = np.where(green_mask)
        
        if len(green_coords[0]) == 0:
            return None
        
        y_min, y_max = green_coords[0].min(), green_coords[0].max()
        x_min, x_max = green_coords[1].min(), green_coords[1].max()
        
        # REDUCED padding to avoid including green boundary
        padding = 5  # Was 10, now smaller
        y_min = max(0, y_min - padding)
        y_max = min(img_array.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(img_array.shape[1], x_max + padding)
        
        # Make square
        size = max(y_max - y_min, x_max - x_min)
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2
        
        y_min = max(0, center_y - size // 2)
        y_max = min(img_array.shape[0], center_y + size // 2)
        x_min = max(0, center_x - size // 2)
        x_max = min(img_array.shape[1], center_x + size // 2)
        
        return (y_min, y_max, x_min, x_max)
    
    def crop_to_qr(self, img_array: np.ndarray, 
                   bounds: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop image to QR code region"""
        y_min, y_max, x_min, x_max = bounds
        return img_array[y_min:y_max, x_min:x_max]
    
    def process_images(self) -> None:
        """Load and process all images"""
        images = self.load_images()
        
        print("\n=== Final Refined Filtering (with pyzbar + stricter criteria) ===")
        for filename, img_array in images:
            is_valid, confidence, metrics, qr_detected, qr_value = self.detect_qr_code(
                filename, img_array
            )
            
            qr_number = self.extract_qr_number(filename)
            filename_valid = self.check_filename_valid_range(filename)
            
            detection = QRDetection(
                filename=filename,
                image_array=img_array,
                is_valid=is_valid,
                confidence=confidence,
                metrics=metrics,
                qr_detected=qr_detected,
                qr_value=qr_value
            )
            
            # Try to detect bounds and crop
            bounds = self.detect_green_triangle_bounds(img_array)
            detection.bounding_box = bounds
            
            if bounds is not None and is_valid:
                detection.cropped_image = self.crop_to_qr(img_array, bounds)
            
            self.detections.append(detection)
            
            status = "✓ QR" if is_valid else "✗ GARBAGE"
            qr_marker = " [pyzbar]" if qr_detected else ""
            print(f"{status} ({confidence:.2f}){qr_marker}: {filename} (QR_{qr_number})")
            if not filename_valid:
                print(f"    ✗ Filename out of range (expecting QR_026-049, got QR_{qr_number})")
            print(f"    Edge: {metrics['edge_density']:.4f}, Green: {metrics['green_density']:.4f}, Contrast: {metrics['contrast']:.1f}")
            if qr_value:
                print(f"    QR Value: {qr_value}")
        
        self.valid_qr_images = [d for d in self.detections if d.is_valid and d.cropped_image is not None]
        print(f"\n=== SUMMARY ===")
        print(f"Total images: {len(self.detections)}")
        print(f"Valid QR codes: {len(self.valid_qr_images)}")
        print(f"Garbage images: {len(self.detections) - len(self.valid_qr_images)}")
    
    # NEW: Use sharpest image (no fusion)
    def fuse_qr_images(self) -> Optional[np.ndarray]:
        if not self.valid_qr_images:
            print("No valid QR images to fuse!")
            return None
        
        # Get only high-confidence crops
        high_confidence = [d for d in self.valid_qr_images if d.confidence > 0.6]
        if not high_confidence:
            high_confidence = self.valid_qr_images
        
        crops = [d.cropped_image for d in high_confidence if d.cropped_image is not None]
        
        if not crops:
            return None
        
        # SELECT SHARPEST IMAGE INSTEAD OF AVERAGING
        def calculate_sharpness(img):
            if len(img.shape) == 3:
                gray = np.mean(img, axis=2)
            else:
                gray = img
            # Laplacian-based sharpness
            edges = np.abs(np.diff(gray.flatten()))
            return np.mean(edges > 20)
        
        sharpness_scores = [calculate_sharpness(crop) for crop in crops]
        best_idx = np.argmax(sharpness_scores)
        fused = crops[best_idx]
        
        print(f"Selected sharpest image (index {best_idx}, "
            f"sharpness {sharpness_scores[best_idx]:.4f})")
        
        return fused

    
    def convert_to_binary_array(self, qr_image: np.ndarray) -> np.ndarray:
        """Convert to binary array"""
        if len(qr_image.shape) == 3:
            gray = np.mean(qr_image, axis=2)
        else:
            gray = qr_image.astype(float)
        
        threshold = np.mean(gray)
        binary = (gray < threshold).astype(int)
        
        return binary
    
    def decode_qr_array(self, binary_array: np.ndarray) -> Dict:
        """Decode QR array"""
        h, w = binary_array.shape
        
        tl_pattern = binary_array[:7, :7] if h >= 7 and w >= 7 else np.array([])
        tr_pattern = binary_array[:7, -7:] if h >= 7 and w >= 7 else np.array([])
        bl_pattern = binary_array[-7:, :7] if h >= 7 and w >= 7 else np.array([])
        
        timing_h = binary_array[6, 8:-8] if h > 6 else np.array([])
        timing_v = binary_array[8:-8, 6] if w > 6 else np.array([])
        
        data_modules = np.sum(binary_array)
        
        result = {
            'size': (h, w),
            'total_modules': h * w,
            'data_modules': int(data_modules),
            'module_density': float(data_modules / (h * w)) if (h * w) > 0 else 0,
            'corner_patterns': {
                'top_left': tl_pattern.tolist() if tl_pattern.size > 0 else [],
                'top_right': tr_pattern.tolist() if tr_pattern.size > 0 else [],
                'bottom_left': bl_pattern.tolist() if bl_pattern.size > 0 else []
            },
            'timing_patterns': {
                'horizontal': timing_h.tolist() if len(timing_h) > 0 else [],
                'vertical': timing_v.tolist() if len(timing_v) > 0 else []
            }
        }
        
        return result
    
    def save_results(self, qr_array: np.ndarray, decode_info: Dict) -> None:
        """Save results"""
        csv_path = os.path.join(self.output_dir, "qr_code_array.csv")
        np.savetxt(csv_path, qr_array, delimiter=',', fmt='%d')
        print(f"✓ Binary array saved: {csv_path}")
        
        npz_path = os.path.join(self.output_dir, "qr_code_array.npz")
        np.savez_compressed(npz_path, qr_array=qr_array)
        print(f"✓ Binary array saved (compressed): {npz_path}")
        
        json_path = os.path.join(self.output_dir, "qr_decode_info.json")
        with open(json_path, 'w') as f:
            json.dump(decode_info, f, indent=2)
        print(f"✓ Decode info saved: {json_path}")
        
        report_path = os.path.join(self.output_dir, "processing_report.json")
        report = {
            'total_images_processed': len(self.detections),
            'valid_qr_codes': len(self.valid_qr_images),
            'garbage_images': len(self.detections) - len(self.valid_qr_images),
            'fusion_source_count': len([d for d in self.valid_qr_images if d.cropped_image is not None]),
            'pyzbar_available': PYZBAR_AVAILABLE,
            'valid_qr_range': self.valid_qr_range,
            'detection_criteria': {
                'filename_range': f"QR_{self.valid_qr_range[0]:03d}-{self.valid_qr_range[1]:03d}",
                'edge_density_min': 0.010,
                'green_density_min': 0.035,
                'contrast_max': 70,
                'pattern_variance_min': 500
            },
            'detections': [
                {
                    'filename': d.filename,
                    'qr_number': self.extract_qr_number(d.filename),
                    'is_valid': d.is_valid,
                    'qr_detected': d.qr_detected,
                    'qr_value': d.qr_value,
                    'confidence': float(d.confidence),
                    'has_bounds': d.bounding_box is not None,
                    'metrics': {
                        'edge_density': d.metrics['edge_density'],
                        'green_density': d.metrics['green_density'],
                        'contrast': d.metrics['contrast'],
                        'pattern_variance': d.metrics['pattern_variance'],
                        'dark_corners': d.metrics['dark_corners']
                    }
                }
                for d in self.detections
            ]
        }
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Processing report saved: {report_path}")
        
        crop_dir = os.path.join(self.output_dir, "cropped_images")
        os.makedirs(crop_dir, exist_ok=True)
        
        for i, detection in enumerate(self.valid_qr_images):
            if detection.cropped_image is not None:
                crop_path = os.path.join(crop_dir, 
                                        f"qr_crop_{i:03d}_{detection.filename}")
                Image.fromarray(detection.cropped_image).save(crop_path)
        
        print(f"✓ Individual crops saved to: {crop_dir}")
    
    def run(self) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Run complete pipeline"""
        print("=== Final Refined QR Code Processing Pipeline ===\n")
        
        self.process_images()
        
        fused = self.fuse_qr_images()
        
        if fused is None:
            print("\n⚠ No valid QR codes to process!")
            return None, None
        
        print("\n=== Converting to Binary Array ===")
        binary_array = self.convert_to_binary_array(fused)
        print(f"Binary array shape: {binary_array.shape}")
        
        print("\n=== Analyzing QR Code Structure ===")
        decode_info = self.decode_qr_array(binary_array)
        print(f"QR Code size: {decode_info['size']}")
        print(f"Module density: {decode_info['module_density']:.2%}")
        
        print("\n=== Saving Results ===")
        self.save_results(binary_array, decode_info)
        
        return binary_array, decode_info


if __name__ == "__main__":
    SOURCE_FOLDER = r"C:\Users\anuj_\Downloads\TeamCANDY\qr_detections"
    OUTPUT_FOLDER = r"C:\Users\anuj_\Downloads\TeamCANDY\qr_processed_final"
    
    processor = FinalRefinedQRCodeProcessor(
        SOURCE_FOLDER, 
        OUTPUT_FOLDER,
        valid_qr_range=(26, 49)  # Only QR_026 through QR_049
    )
    
    binary_array, decode_info = processor.run()
    
    if binary_array is not None:
        print("\n=== SUCCESS ===")
        print(f"QR Code binary array shape: {binary_array.shape}")
        print(f"Output saved to: {OUTPUT_FOLDER}")
        print("\nBinary Array Preview (20x20):")
        print(binary_array[:20, :20])
    else:
        print("\n⚠ Processing completed with no valid QR codes found")