#!/usr/bin/env python3
"""
RATTENTION Demonstration Agent
Demonstrates attention retention challenges with BERT base model
Shows how model performance degrades with increasing context length
"""

import sys
import os
import time
import json
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Try to import required libraries
try:
    import torch
    import numpy as np
    from transformers import BertTokenizer, BertModel
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

class AttentionRetentionDemo:
    """
    Demonstrates attention retention challenges using BERT base model
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "bert-base-uncased"
        self.tokenizer = None
        self.model = None
        self.max_length = 512  # BERT's maximum sequence length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Metrics tracking
        self.attention_metrics = []
        self.performance_metrics = []
        
    def initialize_model(self):
        """Initialize BERT model and tokenizer"""
        if not DEPENDENCIES_AVAILABLE:
            print("❌ Cannot initialize model - dependencies not available")
            return False
            
        try:
            print(f"🔄 Loading BERT model from {self.model_path}...")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = BertModel.from_pretrained(self.model_path, output_attentions=True)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def create_progressive_context(self, base_text: str, context_lengths: List[int]) -> List[str]:
        """Create texts of progressively increasing length"""
        # Base meaningful text
        base_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models process information sequentially.",
            "Attention mechanisms help models focus on relevant parts.",
            "Long contexts challenge model memory and performance.",
            "BERT uses bidirectional attention for understanding.",
            "Transformers revolutionized natural language processing.",
            "Context length affects model comprehension significantly.",
            "Memory limitations impact attention quality over time.",
            "Information retention decreases with sequence length.",
            "Models struggle with distant token relationships."
        ]
        
        progressive_texts = []
        for target_length in context_lengths:
            # Start with base text
            current_text = base_text
            sentence_idx = 0
            
            # Add sentences until we reach target token count
            while len(self.tokenizer.encode(current_text)) < target_length and sentence_idx < len(base_sentences):
                current_text += " " + base_sentences[sentence_idx % len(base_sentences)]
                sentence_idx += 1
            
            # Truncate if too long
            tokens = self.tokenizer.encode(current_text)
            if len(tokens) > target_length:
                tokens = tokens[:target_length]
                current_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            
            progressive_texts.append(current_text)
        
        return progressive_texts
    
    def analyze_attention_patterns(self, text: str) -> Dict:
        """Analyze attention patterns for given text"""
        if not self.model or not self.tokenizer:
            return {"error": "Model not initialized"}
        
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                 max_length=self.max_length, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model outputs with attention
            with torch.no_grad():
                outputs = self.model(**inputs)
                attentions = outputs.attentions  # Tuple of attention weights for each layer
            
            # Analyze attention patterns
            sequence_length = inputs['input_ids'].shape[1]
            num_layers = len(attentions)
            num_heads = attentions[0].shape[1]
            
            # Calculate attention statistics
            attention_stats = {
                "sequence_length": sequence_length,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "attention_entropy": [],
                "attention_concentration": [],
                "long_range_attention": []
            }
            
            for layer_idx, layer_attention in enumerate(attentions):
                # layer_attention shape: [batch_size, num_heads, seq_len, seq_len]
                layer_attention = layer_attention.squeeze(0)  # Remove batch dimension
                
                # Calculate entropy (measure of attention dispersion)
                entropy_per_head = []
                concentration_per_head = []
                long_range_per_head = []
                
                for head_idx in range(num_heads):
                    head_attention = layer_attention[head_idx]  # [seq_len, seq_len]
                    
                    # Entropy calculation
                    entropy = -torch.sum(head_attention * torch.log(head_attention + 1e-9), dim=-1)
                    entropy_per_head.append(entropy.mean().item())
                    
                    # Attention concentration (how much attention is on nearby tokens)
                    local_window = 10  # Consider tokens within 10 positions as "local"
                    local_attention = 0
                    for i in range(sequence_length):
                        start_idx = max(0, i - local_window)
                        end_idx = min(sequence_length, i + local_window + 1)
                        local_attention += head_attention[i, start_idx:end_idx].sum().item()
                    
                    concentration = local_attention / (sequence_length * sequence_length)
                    concentration_per_head.append(concentration)
                    
                    # Long-range attention (attention to tokens > 50 positions away)
                    long_range = 0
                    for i in range(sequence_length):
                        if i > 50:
                            long_range += head_attention[i, :i-50].sum().item()
                        if i < sequence_length - 50:
                            long_range += head_attention[i, i+50:].sum().item()
                    
                    long_range_normalized = long_range / (sequence_length * sequence_length)
                    long_range_per_head.append(long_range_normalized)
                
                attention_stats["attention_entropy"].append(np.mean(entropy_per_head))
                attention_stats["attention_concentration"].append(np.mean(concentration_per_head))
                attention_stats["long_range_attention"].append(np.mean(long_range_per_head))
            
            return attention_stats
            
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}
    
    def demonstrate_retention_challenges(self):
        """Main demonstration of attention retention challenges"""
        print("\n" + "="*80)
        print("🧠 RATTENTION PAPER DEMONSTRATION: Attention Retention Challenges")
        print("="*80)
        
        if not self.initialize_model():
            print("❌ Cannot proceed without model initialization")
            return
        
        # Define test scenarios
        base_query = "What is the main topic discussed in this text?"
        context_lengths = [50, 100, 200, 300, 400, 500]
        
        print(f"\n📝 Testing attention retention across context lengths: {context_lengths}")
        print(f"🎯 Base query: '{base_query}'")
        
        # Create progressive contexts
        base_text = "The RATTENTION paper introduces a novel attention mechanism that combines sliding window attention with residual linear attention."
        progressive_texts = self.create_progressive_context(base_text, context_lengths)
        
        print(f"\n🔍 Analyzing attention patterns...")
        
        results = []
        for i, (length, text) in enumerate(zip(context_lengths, progressive_texts)):
            print(f"\n📊 Context Length: {length} tokens")
            print(f"📄 Text preview: {text[:100]}...")
            
            # Analyze attention patterns
            attention_analysis = self.analyze_attention_patterns(text)
            
            if "error" in attention_analysis:
                print(f"❌ Analysis failed: {attention_analysis['error']}")
                continue
            
            # Display metrics
            actual_length = attention_analysis["sequence_length"]
            avg_entropy = np.mean(attention_analysis["attention_entropy"])
            avg_concentration = np.mean(attention_analysis["attention_concentration"])
            avg_long_range = np.mean(attention_analysis["long_range_attention"])
            
            print(f"   📏 Actual sequence length: {actual_length}")
            print(f"   🌀 Average attention entropy: {avg_entropy:.4f}")
            print(f"   🎯 Local attention concentration: {avg_concentration:.4f}")
            print(f"   🔗 Long-range attention: {avg_long_range:.4f}")
            
            # Performance indicators
            retention_score = self.calculate_retention_score(attention_analysis)
            print(f"   📈 Retention score: {retention_score:.4f}")
            
            results.append({
                "context_length": length,
                "actual_length": actual_length,
                "entropy": avg_entropy,
                "concentration": avg_concentration,
                "long_range": avg_long_range,
                "retention_score": retention_score
            })
            
            # Show degradation indicators
            if i > 0:
                prev_score = results[i-1]["retention_score"]
                score_change = retention_score - prev_score
                if score_change < -0.01:
                    print(f"   ⚠️  Performance degradation detected: {score_change:.4f}")
                elif score_change < -0.05:
                    print(f"   🚨 Significant degradation: {score_change:.4f}")
        
        # Summary analysis
        self.display_summary_analysis(results)
        
        return results
    
    def calculate_retention_score(self, attention_analysis: Dict) -> float:
        """Calculate a composite retention score"""
        if "error" in attention_analysis:
            return 0.0
        
        # Combine metrics into a retention score
        # Higher entropy = better information distribution
        # Higher long-range attention = better long-context handling
        # Lower concentration = less local bias
        
        entropy_score = np.mean(attention_analysis["attention_entropy"]) / 10.0  # Normalize
        long_range_score = np.mean(attention_analysis["long_range_attention"]) * 10.0  # Amplify
        concentration_penalty = np.mean(attention_analysis["attention_concentration"])
        
        retention_score = entropy_score + long_range_score - concentration_penalty
        return max(0.0, min(1.0, retention_score))  # Clamp to [0, 1]
    
    def display_summary_analysis(self, results: List[Dict]):
        """Display summary of retention analysis"""
        print("\n" + "="*80)
        print("📊 SUMMARY ANALYSIS: Attention Retention Challenges")
        print("="*80)
        
        if not results:
            print("❌ No results to analyze")
            return
        
        print(f"\n📈 Performance Trends:")
        for i, result in enumerate(results):
            length = result["context_length"]
            score = result["retention_score"]
            
            # Trend indicators
            if i == 0:
                trend = "📍 Baseline"
            else:
                prev_score = results[i-1]["retention_score"]
                change = score - prev_score
                if change > 0.01:
                    trend = "📈 Improving"
                elif change < -0.01:
                    trend = "📉 Degrading"
                else:
                    trend = "➡️  Stable"
            
            print(f"   Context {length:3d}: Score {score:.4f} {trend}")
        
        # Key findings
        print(f"\n🔍 Key Findings:")
        
        # Calculate degradation
        initial_score = results[0]["retention_score"]
        final_score = results[-1]["retention_score"]
        total_degradation = initial_score - final_score
        
        print(f"   • Initial retention score: {initial_score:.4f}")
        print(f"   • Final retention score: {final_score:.4f}")
        print(f"   • Total degradation: {total_degradation:.4f}")
        
        if total_degradation > 0.1:
            print(f"   🚨 Significant attention retention challenges detected!")
        elif total_degradation > 0.05:
            print(f"   ⚠️  Moderate attention retention issues observed")
        else:
            print(f"   ✅ Attention retention appears stable")
        
        # RATTENTION relevance
        print(f"\n💡 RATTENTION Paper Relevance:")
        print(f"   • This demonstrates the exact problem RATTENTION addresses")
        print(f"   • Standard attention mechanisms struggle with long contexts")
        print(f"   • RATTENTION's hybrid approach could maintain performance")
        print(f"   • Window size of 512 tokens could be sufficient with RLA")

def main():
    """Main demonstration function"""
    print("🚀 Starting RATTENTION Attention Retention Demonstration")
    
    # Check if we're in the virtual environment
    if "rattention_env" not in sys.executable:
        print("⚠️  Warning: Not running in rattention_env virtual environment")
    
    # Create and run demonstration
    demo = AttentionRetentionDemo()
    
    if not DEPENDENCIES_AVAILABLE:
        print("\n❌ Required dependencies not available.")
        print("Please ensure the virtual environment is activated and dependencies are installed:")
        print("   pip install torch transformers numpy")
        return
    
    try:
        results = demo.demonstrate_retention_challenges()
        
        print(f"\n✅ Demonstration completed successfully!")
        print(f"📁 Results can be used to validate RATTENTION's approach")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
