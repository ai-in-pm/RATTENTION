#!/usr/bin/env python3
"""
RATTENTION Test Cases
Example test cases demonstrating attention retention problems
"""

import json
from typing import List, Dict, Tuple

class RattentionTestCases:
    """
    Collection of test cases that demonstrate attention retention challenges
    """
    
    def __init__(self):
        self.test_cases = self._create_test_cases()
    
    def _create_test_cases(self) -> List[Dict]:
        """Create comprehensive test cases"""
        
        test_cases = [
            {
                "name": "Information Retrieval Challenge",
                "description": "Tests ability to retrieve specific information from early in the context",
                "base_info": "The secret code is ALPHA-7829. Remember this code for later reference.",
                "context_builders": [
                    "This document contains important security protocols for the organization.",
                    "All employees must follow the established guidelines for data protection.",
                    "Regular training sessions are conducted to ensure compliance with policies.",
                    "The IT department maintains strict access controls for sensitive systems.",
                    "Backup procedures are implemented to prevent data loss incidents.",
                    "Network security measures include firewalls and intrusion detection systems.",
                    "User authentication requires multi-factor verification for enhanced security.",
                    "Incident response teams are available 24/7 for emergency situations.",
                    "Regular audits ensure compliance with industry standards and regulations.",
                    "Data encryption is mandatory for all confidential information transfers."
                ],
                "query": "What was the secret code mentioned at the beginning?",
                "expected_challenge": "Model should struggle to recall ALPHA-7829 as context grows"
            },
            
            {
                "name": "Conversation Context Tracking",
                "description": "Tests ability to maintain conversation context across multiple exchanges",
                "base_info": "User: My name is Sarah and I work as a data scientist at TechCorp. I'm interested in machine learning applications for healthcare.",
                "context_builders": [
                    "Assistant: That's fascinating! Healthcare is a great domain for ML applications.",
                    "User: Yes, I'm particularly interested in diagnostic imaging and pattern recognition.",
                    "Assistant: Computer vision techniques have shown remarkable results in medical imaging.",
                    "User: Exactly! We're working on a project to detect early signs of diseases in X-rays.",
                    "Assistant: That sounds like important work. What specific diseases are you focusing on?",
                    "User: Primarily lung cancer and pneumonia detection using deep learning models.",
                    "Assistant: Those are critical applications. Have you considered using attention mechanisms?",
                    "User: We've experimented with some, but we're facing challenges with long sequences.",
                    "Assistant: Long sequence modeling is indeed challenging. What's your current approach?",
                    "User: We're using standard transformers, but they struggle with high-resolution images."
                ],
                "query": "What is the user's name and company?",
                "expected_challenge": "Model should lose track of Sarah/TechCorp as conversation continues"
            },
            
            {
                "name": "Multi-Topic Document Analysis",
                "description": "Tests ability to maintain awareness of multiple topics in a long document",
                "base_info": "TOPIC A: Climate change impacts on agriculture. TOPIC B: Renewable energy solutions. TOPIC C: Urban planning strategies.",
                "context_builders": [
                    "Climate change significantly affects crop yields and farming practices worldwide.",
                    "Rising temperatures and changing precipitation patterns challenge traditional agriculture.",
                    "Solar and wind energy technologies have become increasingly cost-effective solutions.",
                    "Energy storage systems are crucial for managing renewable energy intermittency.",
                    "Smart city initiatives integrate technology to improve urban living conditions.",
                    "Sustainable transportation systems reduce environmental impact in cities.",
                    "Drought-resistant crops are being developed to adapt to changing climate conditions.",
                    "Grid modernization enables better integration of renewable energy sources.",
                    "Urban green spaces provide environmental and social benefits to communities.",
                    "Precision agriculture uses data analytics to optimize farming efficiency."
                ],
                "query": "List all three main topics discussed in this document.",
                "expected_challenge": "Model should struggle to remember all three topics as content grows"
            },
            
            {
                "name": "Numerical Sequence Tracking",
                "description": "Tests ability to track numerical information across long contexts",
                "base_info": "Initial values: A=100, B=250, C=75, D=180, E=320.",
                "context_builders": [
                    "The quarterly report shows significant growth in all business segments.",
                    "Revenue increased by 15% compared to the previous quarter's performance.",
                    "Customer satisfaction scores improved across all product categories.",
                    "Market expansion efforts resulted in new partnerships and opportunities.",
                    "Operational efficiency gains were achieved through process optimization.",
                    "Technology investments enhanced productivity and reduced operational costs.",
                    "Employee training programs contributed to improved service quality metrics.",
                    "Supply chain improvements reduced delivery times and inventory costs.",
                    "Research and development initiatives led to innovative product features.",
                    "Competitive analysis revealed opportunities for market differentiation."
                ],
                "query": "What were the initial values of A, B, C, D, and E?",
                "expected_challenge": "Model should lose track of specific numerical values"
            },
            
            {
                "name": "Temporal Sequence Challenge",
                "description": "Tests ability to maintain temporal ordering of events",
                "base_info": "Timeline: 9:00 AM - Project kickoff meeting. 10:30 AM - Requirements gathering. 12:00 PM - Lunch break.",
                "context_builders": [
                    "The development team discussed technical architecture and implementation strategies.",
                    "Stakeholders provided feedback on user interface mockups and design concepts.",
                    "Quality assurance procedures were established for testing and validation phases.",
                    "Resource allocation was planned to ensure adequate staffing for all project phases.",
                    "Risk assessment identified potential challenges and mitigation strategies.",
                    "Communication protocols were established for regular progress updates.",
                    "Budget considerations were reviewed to ensure project financial viability.",
                    "Timeline milestones were defined with specific deliverables and deadlines.",
                    "Technology stack decisions were made based on project requirements.",
                    "Documentation standards were established for code and process documentation."
                ],
                "query": "What happened at 10:30 AM according to the timeline?",
                "expected_challenge": "Model should struggle to recall specific timeline details"
            }
        ]
        
        return test_cases
    
    def generate_progressive_contexts(self, test_case: Dict, context_lengths: List[int]) -> List[Tuple[str, str]]:
        """Generate progressive contexts for a test case"""
        base_info = test_case["base_info"]
        builders = test_case["context_builders"]
        query = test_case["query"]
        
        contexts = []
        
        for target_length in context_lengths:
            # Start with base info
            current_context = base_info
            builder_idx = 0
            
            # Add context builders until we reach target length (approximately)
            while len(current_context.split()) < target_length and builder_idx < len(builders):
                current_context += " " + builders[builder_idx % len(builders)]
                builder_idx += 1
            
            # Add some padding if needed
            while len(current_context.split()) < target_length:
                current_context += " Additional context information to reach target length."
            
            contexts.append((current_context, query))
        
        return contexts
    
    def get_test_case(self, name: str) -> Dict:
        """Get a specific test case by name"""
        for test_case in self.test_cases:
            if test_case["name"] == name:
                return test_case
        return None
    
    def list_test_cases(self) -> List[str]:
        """List all available test case names"""
        return [tc["name"] for tc in self.test_cases]
    
    def run_test_case_demo(self, test_case_name: str, context_lengths: List[int] = None):
        """Run a demonstration of a specific test case"""
        if context_lengths is None:
            context_lengths = [50, 100, 200, 300, 400]
        
        test_case = self.get_test_case(test_case_name)
        if not test_case:
            print(f"❌ Test case '{test_case_name}' not found")
            return
        
        print(f"\n🧪 Test Case: {test_case['name']}")
        print(f"📝 Description: {test_case['description']}")
        print(f"🎯 Expected Challenge: {test_case['expected_challenge']}")
        print("="*80)
        
        contexts = self.generate_progressive_contexts(test_case, context_lengths)
        
        for i, (context, query) in enumerate(contexts):
            length = context_lengths[i]
            word_count = len(context.split())
            
            print(f"\n📊 Context Length: ~{length} words (actual: {word_count})")
            print(f"📄 Context Preview: {context[:100]}...")
            print(f"❓ Query: {query}")
            print(f"💭 Expected Behavior: As context grows, model should struggle more with this query")
            
            # Simulate attention degradation
            attention_quality = max(0.1, 1.0 - (length / 500.0))
            if attention_quality < 0.7:
                print(f"⚠️  Simulated attention quality: {attention_quality:.2f} - Degradation likely")
            elif attention_quality < 0.5:
                print(f"🚨 Simulated attention quality: {attention_quality:.2f} - Significant degradation")
            else:
                print(f"✅ Simulated attention quality: {attention_quality:.2f} - Good retention")
    
    def export_test_cases(self, filename: str = "rattention_test_cases.json"):
        """Export test cases to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.test_cases, f, indent=2, ensure_ascii=False)
            print(f"✅ Test cases exported to {filename}")
        except Exception as e:
            print(f"❌ Failed to export test cases: {e}")

def main():
    """Demonstrate the test cases"""
    print("🧪 RATTENTION Test Cases Demonstration")
    print("="*60)
    
    test_suite = RattentionTestCases()
    
    print(f"\n📋 Available Test Cases:")
    for i, name in enumerate(test_suite.list_test_cases(), 1):
        print(f"   {i}. {name}")
    
    # Run a sample test case
    print(f"\n🚀 Running Sample Test Case...")
    test_suite.run_test_case_demo("Information Retrieval Challenge")
    
    # Export test cases
    test_suite.export_test_cases()
    
    print(f"\n💡 Usage:")
    print(f"   • These test cases can be used with the main demonstration agent")
    print(f"   • Each case targets specific attention retention challenges")
    print(f"   • Progressive context lengths show degradation patterns")
    print(f"   • Results validate RATTENTION's approach to the problem")

if __name__ == "__main__":
    main()
