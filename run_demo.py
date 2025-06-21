#!/usr/bin/env python3
"""
RATTENTION Demo Runner
Simple CLI interface for running the RATTENTION demonstration
"""

import sys
import os
import argparse
from pathlib import Path

def check_environment():
    """Check if we're in the correct environment"""
    print("🔍 Checking environment...")
    
    # Check if we're in virtual environment
    if "rattention_env" in sys.executable:
        print("✅ Running in rattention_env virtual environment")
    else:
        print("⚠️  Warning: Not in rattention_env virtual environment")
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check for required files
    required_files = [
        "rattention_demo_agent.py",
        "rattention_paper_analysis.md",
        "rattention_paper_content.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print(f"✅ Found: {file}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    return True

def check_dependencies():
    """Check if required dependencies are available"""
    print("\n🔍 Checking dependencies...")
    
    dependencies = {
        "torch": "PyTorch for deep learning",
        "transformers": "Hugging Face Transformers for BERT",
        "numpy": "NumPy for numerical operations"
    }
    
    available = {}
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            print(f"✅ {dep}: {description}")
            available[dep] = True
        except ImportError:
            print(f"❌ {dep}: {description} - NOT AVAILABLE")
            available[dep] = False
    
    all_available = all(available.values())
    
    if not all_available:
        print(f"\n⚠️  Some dependencies are missing. To install:")
        print(f"   .\rattention_env\Scripts\Activate.ps1")
        print(f"   pip install torch transformers numpy")
    
    return all_available

def run_basic_demo():
    """Run the basic attention retention demonstration"""
    print("\n🚀 Starting Basic RATTENTION Demonstration...")
    
    try:
        from rattention_demo_agent import AttentionRetentionDemo
        
        demo = AttentionRetentionDemo()
        results = demo.demonstrate_retention_challenges()
        
        if results:
            print(f"\n✅ Demo completed with {len(results)} test cases")
            return True
        else:
            print(f"\n❌ Demo failed to produce results")
            return False
            
    except ImportError as e:
        print(f"❌ Cannot import demo module: {e}")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def run_simple_demo():
    """Run a simplified demo without heavy dependencies"""
    print("\n🚀 Starting Simplified RATTENTION Demonstration...")
    print("="*60)
    
    # Simulate attention retention challenges
    context_lengths = [50, 100, 200, 300, 400, 500]
    
    print("📊 Simulated Attention Retention Analysis")
    print("(This demo simulates the behavior when dependencies are unavailable)")
    
    for i, length in enumerate(context_lengths):
        # Simulate degrading performance
        base_score = 0.85
        degradation_factor = length / 1000.0  # Simulate degradation
        retention_score = max(0.1, base_score - degradation_factor)
        
        print(f"\n📏 Context Length: {length} tokens")
        print(f"   📈 Simulated retention score: {retention_score:.4f}")
        
        if i > 0:
            prev_score = max(0.1, base_score - context_lengths[i-1] / 1000.0)
            change = retention_score - prev_score
            if change < -0.05:
                print(f"   🚨 Significant degradation: {change:.4f}")
            elif change < -0.01:
                print(f"   ⚠️  Performance degradation: {change:.4f}")
    
    print(f"\n💡 Key Insights (Simulated):")
    print(f"   • Attention retention decreases with context length")
    print(f"   • Standard BERT struggles beyond ~300 tokens effectively")
    print(f"   • RATTENTION's 512-token window could address this")
    print(f"   • Hybrid attention mechanisms show promise")
    
    return True

def display_paper_summary():
    """Display a summary of the RATTENTION paper"""
    print("\n📄 RATTENTION Paper Summary")
    print("="*60)
    
    try:
        with open("rattention_paper_analysis.md", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract key sections
        lines = content.split("\n")
        in_summary = False
        summary_lines = []
        
        for line in lines:
            if "## Executive Summary" in line:
                in_summary = True
                continue
            elif line.startswith("## ") and in_summary:
                break
            elif in_summary and line.strip():
                summary_lines.append(line)
        
        if summary_lines:
            print("\n".join(summary_lines[:10]))  # First 10 lines
            print("\n... (see rattention_paper_analysis.md for full analysis)")
        else:
            print("📄 Full analysis available in rattention_paper_analysis.md")
            
    except FileNotFoundError:
        print("❌ Paper analysis file not found")
    except Exception as e:
        print(f"❌ Error reading paper analysis: {e}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="RATTENTION Demonstration Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py --check          # Check environment and dependencies
  python run_demo.py --demo           # Run full demonstration
  python run_demo.py --simple         # Run simplified demo
  python run_demo.py --paper          # Show paper summary
  python run_demo.py --all            # Run everything
        """
    )
    
    parser.add_argument("--check", action="store_true", 
                       help="Check environment and dependencies")
    parser.add_argument("--demo", action="store_true",
                       help="Run the full attention retention demonstration")
    parser.add_argument("--simple", action="store_true",
                       help="Run simplified demonstration (no dependencies required)")
    parser.add_argument("--paper", action="store_true",
                       help="Display RATTENTION paper summary")
    parser.add_argument("--all", action="store_true",
                       help="Run all demonstrations")
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    print("🧠 RATTENTION Paper Demonstration Suite")
    print("="*60)
    
    # Check environment
    if args.check or args.all:
        env_ok = check_environment()
        deps_ok = check_dependencies()
        
        if not env_ok:
            print("❌ Environment check failed")
            return
    else:
        deps_ok = True  # Assume OK if not checking
    
    # Show paper summary
    if args.paper or args.all:
        display_paper_summary()
    
    # Run demonstrations
    if args.demo or args.all:
        if deps_ok:
            success = run_basic_demo()
            if not success:
                print("⚠️  Falling back to simplified demo...")
                run_simple_demo()
        else:
            print("⚠️  Dependencies not available, running simplified demo...")
            run_simple_demo()
    
    elif args.simple:
        run_simple_demo()
    
    print(f"\n✅ Demonstration suite completed!")
    print(f"📁 Check the following files for more details:")
    print(f"   • rattention_paper_analysis.md - Comprehensive paper analysis")
    print(f"   • rattention_paper_content.txt - Full paper text")
    print(f"   • rattention_demo_agent.py - Demonstration code")

if __name__ == "__main__":
    main()
