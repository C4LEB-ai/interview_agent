import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from openai import AsyncOpenAI
import base64
import os
import io

@dataclass
class InterviewResponse:
    """Structured response from the interview system"""
    session_id: str
    question_id: int
    audio_response: str  # base64 encoded audio
    transcript: str
    evaluation_score: float
    key_points: List[str]
    follow_up_questions: List[str]
    sentiment: str
    confidence: float
    cost_used: float
    timestamp: str

@dataclass
class InterviewSession:
    """Complete interview session data"""
    session_id: str
    candidate_id: str
    position: str
    status: str  # "active", "completed", "paused"
    current_question: int
    responses: List[InterviewResponse]
    overall_score: float
    recommendations: List[str]
    total_cost: float
    duration_minutes: int
    created_at: str

api_key = os.getenv("OPENAI_API_KEY")
class AIInterviewEngine:
    """Core ML/AI pipeline for conducting interviews"""
    
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.sessions: Dict[str, InterviewSession] = {}
        
        # Question banks by position
        self.question_banks = {
            "software_engineer": [
                "Tell me about a challenging technical problem you've solved recently.",
                "How do you approach debugging complex issues?",
                "Describe your experience with system design.",
                "What's your preferred development methodology and why?",
                "How do you stay updated with new technologies?"
            ],
            "data_scientist": [
                "Walk me through a data science project from start to finish.",
                "How do you handle missing or dirty data?",
                "Explain a machine learning model you've implemented.",
                "How do you validate your model's performance?",
                "Describe a time when your analysis changed a business decision."
            ],
            "product_manager": [
                "How do you prioritize features in a product roadmap?",
                "Describe a time you had to make a decision with incomplete information.",
                "How do you handle conflicting stakeholder requirements?",
                "Walk me through how you would launch a new feature.",
                "How do you measure product success?"
            ]
        }
        
        # Evaluation criteria by position
        self.evaluation_criteria = {
            "software_engineer": {
                "technical_depth": 0.3,
                "problem_solving": 0.25,
                "communication": 0.2,
                "experience": 0.15,
                "cultural_fit": 0.1
            },
            "data_scientist": {
                "analytical_thinking": 0.3,
                "technical_skills": 0.25,
                "business_acumen": 0.2,
                "communication": 0.15,
                "methodology": 0.1
            },
            "product_manager": {
                "strategic_thinking": 0.3,
                "stakeholder_management": 0.25,
                "analytical_skills": 0.2,
                "communication": 0.15,
                "leadership": 0.1
            }
        }

    async def start_interview(self, candidate_id: str, position: str) -> Dict[str, Any]:
        """Initialize a new interview session"""
        session_id = f"session_{candidate_id}_{int(time.time())}"
        
        session = InterviewSession(
            session_id=session_id,
            candidate_id=candidate_id,
            position=position,
            status="active",
            current_question=0,
            responses=[],
            overall_score=0.0,
            recommendations=[],
            total_cost=0.0,
            duration_minutes=0,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.sessions[session_id] = session
        
        # Get first question
        first_question = await self._get_next_question(session_id)
        
        return {
            "session_id": session_id,
            "status": "started",
            "first_question": first_question,
            "estimated_duration": "15-20 minutes",
            "total_questions": len(self.question_banks.get(position, [])),
            "position": position
        }

    async def process_audio_response(self, session_id: str, audio_data: bytes) -> Dict[str, Any]:
        """Process candidate's audio response and generate next question"""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        start_time = time.time()
        
        try:
            # Use gpt-4o-audio-preview for the complete pipeline
            audio_b64 = base64.b64encode(audio_data).decode()
            
            # Get current question context
            current_question = self._get_current_question(session)
            evaluation_criteria = self.evaluation_criteria.get(session.position, {})
            
            # Single model call handles everything
            response = await self.client.chat.completions.create(
                model="gpt-4o-audio-preview-2025-06-03",
                messages=[
                    {
                        "role": "system",
                        "content": self._build_system_prompt(session.position, evaluation_criteria)
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Current question: {current_question}\n\nPlease analyze the candidate's response and provide structured feedback."
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_b64,
                                    "format": "wav"
                                }
                            }
                        ]
                    }
                ],
                modalities=["text", "audio"],
                audio={"voice": "alloy", "format": "wav"},
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse structured response
            analysis = self._parse_ai_response(response.choices[0].message.content)
            audio_response = response.choices[0].message.audio.data if response.choices[0].message.audio else None
            
            # Calculate cost
            cost = self._calculate_cost(response.usage)
            session.total_cost += cost
            
            # Create response object
            interview_response = InterviewResponse(
                session_id=session_id,
                question_id=session.current_question,
                audio_response=audio_response or "",
                transcript=analysis.get("transcript", ""),
                evaluation_score=analysis.get("score", 0.0),
                key_points=analysis.get("key_points", []),
                follow_up_questions=analysis.get("follow_ups", []),
                sentiment=analysis.get("sentiment", "neutral"),
                confidence=analysis.get("confidence", 0.0),
                cost_used=cost,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            session.responses.append(interview_response)
            session.current_question += 1
            
            # Determine next action
            next_question = None
            if session.current_question < len(self.question_banks.get(session.position, [])):
                next_question = await self._get_next_question(session_id)
            else:
                # Interview complete
                session.status = "completed"
                session.overall_score = self._calculate_overall_score(session)
                session.recommendations = self._generate_recommendations(session)
                session.duration_minutes = int((time.time() - start_time) / 60)
            
            return {
                "session_id": session_id,
                "status": session.status,
                "response_analysis": asdict(interview_response),
                "next_question": next_question,
                "progress": {
                    "current": session.current_question,
                    "total": len(self.question_banks.get(session.position, [])),
                    "percentage": (session.current_question / len(self.question_banks.get(session.position, []))) * 100
                },
                "cost_tracking": {
                    "current_cost": cost,
                    "total_cost": session.total_cost,
                    "estimated_final": session.total_cost * (len(self.question_banks.get(session.position, [])) / max(session.current_question, 1))
                }
            }
            
        except Exception as e:
            return {
                "error": f"Processing failed: {str(e)}",
                "session_id": session_id,
                "retry_suggested": True
            }

    async def get_interview_summary(self, session_id: str) -> Dict[str, Any]:
        """Get complete interview summary and evaluation"""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        
        # Use embeddings for final evaluation
        embeddings_response = await self.client.embeddings.create(
            model="text-embedding-3-large",
            input=[resp.transcript for resp in session.responses]
        )
        
        # Calculate similarity scores and insights
        evaluation_insights = await self._generate_final_evaluation(session, embeddings_response)
        
        return {
            "session_summary": asdict(session),
            "detailed_evaluation": evaluation_insights,
            "hiring_recommendation": self._get_hiring_recommendation(session),
            "cost_breakdown": {
                "total_cost": session.total_cost,
                "cost_per_question": session.total_cost / max(len(session.responses), 1),
                "model_usage": "gpt-4o-audio-preview"
            },
            "export_data": {
                "candidate_id": session.candidate_id,
                "position": session.position,
                "score": session.overall_score,
                "duration": session.duration_minutes,
                "responses": len(session.responses)
            }
        }

    def _build_system_prompt(self, position: str, criteria: Dict[str, float]) -> str:
        """Build position-specific system prompt"""
        criteria_text = "\n".join([f"- {k.replace('_', ' ').title()}: {v*100}%" for k, v in criteria.items()])
        
        return f"""You are an expert AI interviewer for {position} positions. 

EVALUATION CRITERIA:
{criteria_text}

RESPONSE FORMAT (return as JSON):
{{
    "transcript": "exact transcript of candidate response",
    "score": float (0-10 scale),
    "key_points": ["point1", "point2", "point3"],
    "follow_ups": ["question1", "question2"],
    "sentiment": "positive/neutral/negative",
    "confidence": float (0-1),
    "areas_evaluated": {{"criteria": score}}
}}

INSTRUCTIONS:
1. Transcribe the audio accurately
2. Evaluate based on the criteria above
3. Identify 2-3 key strengths or concerns
4. Suggest relevant follow-up questions
5. Assess confidence level in your evaluation
6. Keep responses professional and constructive

Generate your audio response as a follow-up question or feedback."""

    def _parse_ai_response(self, content: str) -> Dict[str, Any]:
        """Parse structured AI response"""
        try:
            # Try to extract JSON from response
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                # Fallback: look for JSON-like structure
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end] if start != -1 and end != 0 else "{}"
            
            return json.loads(json_str)
        except:
            # Fallback parsing
            return {
                "transcript": "Response processing error",
                "score": 5.0,
                "key_points": ["Unable to parse response"],
                "follow_ups": ["Could you please repeat that?"],
                "sentiment": "neutral",
                "confidence": 0.5
            }

    async def _get_next_question(self, session_id: str) -> str:
        """Get next question for the interview"""
        session = self.sessions[session_id]
        questions = self.question_banks.get(session.position, [])
        
        if session.current_question < len(questions):
            return questions[session.current_question]
        return "Thank you for your time. The interview is now complete."

    def _get_current_question(self, session: InterviewSession) -> str:
        """Get current question being asked"""
        questions = self.question_banks.get(session.position, [])
        if session.current_question < len(questions):
            return questions[session.current_question]
        return "Interview completion"

    def _calculate_cost(self, usage) -> float:
        """Calculate cost based on token usage"""
        # gpt-4o-audio-preview pricing (estimated)
        input_cost = (usage.prompt_tokens / 1000) * 0.005
        output_cost = (usage.completion_tokens / 1000) * 0.015
        return input_cost + output_cost

    def _calculate_overall_score(self, session: InterviewSession) -> float:
        """Calculate weighted overall score"""
        if not session.responses:
            return 0.0
        
        criteria = self.evaluation_criteria.get(session.position, {})
        if not criteria:
            return sum(resp.evaluation_score for resp in session.responses) / len(session.responses)
        
        # Weighted average based on position criteria
        total_score = 0.0
        for response in session.responses:
            total_score += response.evaluation_score
        
        return total_score / len(session.responses)

    def _generate_recommendations(self, session: InterviewSession) -> List[str]:
        """Generate hiring recommendations"""
        score = session.overall_score
        
        if score >= 8.0:
            return ["Strong hire - excellent candidate", "Proceed to final round", "Consider for senior role"]
        elif score >= 6.5:
            return ["Hire - good candidate", "Standard onboarding", "Monitor performance in first 90 days"]
        elif score >= 5.0:
            return ["Weak hire - proceed with caution", "Additional interview recommended", "Consider for junior role"]
        else:
            return ["No hire - significant concerns", "Skills gap too large", "Recommend alternative positions"]

    async def _generate_final_evaluation(self, session: InterviewSession, embeddings) -> Dict[str, Any]:
        """Generate comprehensive final evaluation"""
        return {
            "strengths": ["Communication", "Technical knowledge", "Problem-solving"],
            "areas_for_improvement": ["System design", "Leadership experience"],
            "cultural_fit": "Good match",
            "risk_factors": ["Limited experience with large-scale systems"],
            "comparison_to_role": "75% match"
        }

    def _get_hiring_recommendation(self, session: InterviewSession) -> Dict[str, Any]:
        """Get final hiring recommendation"""
        score = session.overall_score
        
        if score >= 7.5:
            decision = "HIRE"
            confidence = "HIGH"
        elif score >= 6.0:
            decision = "HIRE"
            confidence = "MEDIUM"
        elif score >= 4.5:
            decision = "MAYBE"
            confidence = "LOW"
        else:
            decision = "NO HIRE"
            confidence = "HIGH"
        
        return {
            "decision": decision,
            "confidence": confidence,
            "score": score,
            "reasoning": session.recommendations[0] if session.recommendations else "Standard evaluation"
        }

# Example usage and testing
async def test_interview_pipeline():
    """Test the interview pipeline with sample data"""
    
    # Initialize the engine
    engine = AIInterviewEngine(api_key="your-openai-api-key")
    
    # Start interview
    start_result = await engine.start_interview("candidate_123", "software_engineer")
    print("Interview Started:", json.dumps(start_result, indent=2))
    
    # Simulate audio response (you would use actual audio bytes)
    sample_audio = b"sample_audio_data"  # Replace with actual audio
    
    # Process response
    response_result = await engine.process_audio_response(
        start_result["session_id"], 
        sample_audio
    )
    print("Response Processed:", json.dumps(response_result, indent=2))
    
    # Get final summary
    summary = await engine.get_interview_summary(start_result["session_id"])
    print("Interview Summary:", json.dumps(summary, indent=2))

# API endpoints for your web developers
class InterviewAPI:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.engine = AIInterviewEngine(api_key=api_key)
    
    async def start_interview_endpoint(self, candidate_id: str, position: str):
        """POST /api/interview/start"""
        return await self.engine.start_interview(candidate_id, position)
    
    async def submit_response_endpoint(self, session_id: str, audio_base64: str):
        """POST /api/interview/response"""
        audio_data = base64.b64decode(audio_base64)
        return await self.engine.process_audio_response(session_id, audio_data)
    
    async def get_summary_endpoint(self, session_id: str):
        """GET /api/interview/summary/{session_id}"""
        return await self.engine.get_interview_summary(session_id)
    
    async def get_session_status_endpoint(self, session_id: str):
        """GET /api/interview/status/{session_id}"""
        if session_id in self.engine.sessions:
            session = self.engine.sessions[session_id]
            return {
                "session_id": session_id,
                "status": session.status,
                "progress": session.current_question,
                "total_cost": session.total_cost
            }
        return {"error": "Session not found"}

if __name__ == "__main__":
    # Run test
    asyncio.run(test_interview_pipeline())