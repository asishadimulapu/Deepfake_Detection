#!/usr/bin/env python3
"""
Test Multimodal Fusion System
Demonstrates combined image and audio deepfake detection
"""

import os
import sys
import random

# Add the deepfake_detection module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'deepfake_detection'))

def test_multimodal_detection():
    """Test the multimodal fusion system with sample data"""
    print("🚀 Testing Multimodal Fusion System")
    print("=" * 60)
    
    try:
        from multimodal_fusion import MultimodalFusionDetector
        
        # Initialize the multimodal detector
        detector = MultimodalFusionDetector()
        
        # Test with sample data
        processed_data_path = os.path.join('processed_data', 'faces')
        audio_dataset_path = os.path.join('datasets', 'audio')
        
        # Get sample images
        real_faces_path = os.path.join(processed_data_path, 'real')
        fake_faces_path = os.path.join(processed_data_path, 'fake')
        
        # Get sample audio files
        real_audio_path = os.path.join(audio_dataset_path, 'real')
        fake_audio_path = os.path.join(audio_dataset_path, 'fake')
        
        test_cases = []
        
        # Test case 1: Real image + Real audio
        if os.path.exists(real_faces_path) and os.path.exists(real_audio_path):
            real_images = [f for f in os.listdir(real_faces_path) if f.endswith('.png')]
            real_audio = [f for f in os.listdir(real_audio_path) if f.endswith('.flac')]
            
            if real_images and real_audio:
                test_cases.append({
                    'image': os.path.join(real_faces_path, random.choice(real_images)),
                    'audio': os.path.join(real_audio_path, random.choice(real_audio)),
                    'expected': 'AUTHENTIC'
                })
        
        # Test case 2: Fake image + Fake audio
        if os.path.exists(fake_faces_path) and os.path.exists(fake_audio_path):
            fake_images = [f for f in os.listdir(fake_faces_path) if f.endswith('.png')]
            fake_audio = [f for f in os.listdir(fake_audio_path) if f.endswith('.flac')]
            
            if fake_images and fake_audio:
                test_cases.append({
                    'image': os.path.join(fake_faces_path, random.choice(fake_images)),
                    'audio': os.path.join(fake_audio_path, random.choice(fake_audio)),
                    'expected': 'DEEPFAKE'
                })
        
        # Test case 3: Image only
        if os.path.exists(real_faces_path):
            real_images = [f for f in os.listdir(real_faces_path) if f.endswith('.png')]
            if real_images:
                test_cases.append({
                    'image': os.path.join(real_faces_path, random.choice(real_images)),
                    'audio': None,
                    'expected': 'AUTHENTIC'
                })
        
        # Test case 4: Audio only
        if os.path.exists(fake_audio_path):
            fake_audio = [f for f in os.listdir(fake_audio_path) if f.endswith('.flac')]
            if fake_audio:
                test_cases.append({
                    'image': None,
                    'audio': os.path.join(fake_audio_path, random.choice(fake_audio)),
                    'expected': 'DEEPFAKE'
                })
        
        if not test_cases:
            print("❌ No test data available")
            print("   Please ensure you have processed face images and audio files")
            return
        
        print(f"🧪 Testing {len(test_cases)} multimodal scenarios...")
        print("-" * 60)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 Test Case {i}:")
            print(f"   Expected: {test_case['expected']}")
            
            if test_case['image']:
                print(f"   Image: {os.path.basename(test_case['image'])}")
            if test_case['audio']:
                print(f"   Audio: {os.path.basename(test_case['audio'])}")
            
            # Run multimodal detection
            try:
                prediction, result = detector.detect_deepfake(
                    test_case['image'], 
                    test_case['audio']
                )
                
                print(f"   Result: {result}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 Multimodal fusion testing completed!")
        print("\n💡 Benefits of multimodal fusion:")
        print("   - Higher accuracy than single modality")
        print("   - Robust to modality-specific attacks")
        print("   - Better confidence estimation")
        print("   - Handles missing modalities gracefully")
        
    except ImportError as e:
        print(f"❌ Error importing multimodal fusion: {e}")
    except Exception as e:
        print(f"❌ Error in multimodal testing: {e}")

def test_custom_files():
    """Test with custom files provided by user"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test multimodal fusion with custom files")
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    
    args = parser.parse_args()
    
    if not args.image and not args.audio:
        print("❌ Please provide --image and/or --audio paths")
        return
    
    try:
        from multimodal_fusion import MultimodalFusionDetector
        
        detector = MultimodalFusionDetector()
        prediction, result = detector.detect_deepfake(args.image, args.audio)
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Test with custom files
        test_custom_files()
    else:
        # Test with sample data
        test_multimodal_detection()

if __name__ == "__main__":
    main() 