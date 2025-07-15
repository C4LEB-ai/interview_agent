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
class TalentProfile:
    """Talent profile information for dynamic question generation"""
    role: str
    experience_level: str  # "entry", "mid", "senior", "lead"
    tech_stack: List[str]  # ["Python", "React", "AWS", etc.]
    industry: str  # "fintech", "healthcare", "ecommerce", etc.
    required_skills: List[str]
    soft_skills: List[str]
    years_experience: int
    education_level: str
    certifications: List[str]
    project_types: List[str]  # ["web apps", "mobile apps", "data pipelines", etc.]

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
    talent_profile: TalentProfile
    status: str  # "active", "completed", "paused"
    current_question: int
    questions: List[str]  # Generated questions for this session
    responses: List[InterviewResponse]
    overall_score: float
    recommendations: List[str]
    total_cost: float
    duration_minutes: int
    created_at: str

class AIInterviewEngine:
    """Core ML/AI pipeline for conducting interviews"""
    
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.sessions: Dict[str, InterviewSession] = {}
        
        # Standard welcome message
        self.welcome_message = "Hi welcome to Nouvells AI engine, my name is Anisa and I am your interviewer. Please could you briefly tell me about yourself?"
        
        # Dynamic evaluation criteria weights by experience level
        self.evaluation_weights = {
            "entry": {
                "technical_knowledge": 0.2,
                "learning_ability": 0.25,
                "communication": 0.2,
                "problem_solving": 0.15,
                "cultural_fit": 0.1,
                "potential": 0.1
            },
            "mid": {
                "technical_depth": 0.3,
                "problem_solving": 0.25,
                "communication": 0.15,
                "experience": 0.15,
                "leadership": 0.1,
                "cultural_fit": 0.05
            },
            "senior": {
                "technical_expertise": 0.25,
                "system_design": 0.2,
                "leadership": 0.2,
                "mentoring": 0.15,
                "strategic_thinking": 0.1,
                "communication": 0.1
            },
            "lead": {
                "technical_vision": 0.2,
                "leadership": 0.25,
                "strategic_thinking": 0.2,
                "team_building": 0.15,
                "business_acumen": 0.1,
                "communication": 0.1
            }
        }

    async def generate_interview_questions(self, talent_profile: TalentProfile) -> List[str]:
        """Generate dynamic interview questions based on talent profile"""
        
        # Create context for question generation
        context = f"""
        Role: {talent_profile.role}
        Experience Level: {talent_profile.experience_level}
        Years of Experience: {talent_profile.years_experience}
        Tech Stack: {', '.join(talent_profile.tech_stack)}
        Industry: {talent_profile.industry}
        Required Skills: {', '.join(talent_profile.required_skills)}
        Soft Skills: {', '.join(talent_profile.soft_skills)}
        Project Types: {', '.join(talent_profile.project_types)}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4.1-2025-04-14",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert interview question generator. Generate 6-8 relevant interview questions based on the talent profile provided. 

                        GUIDELINES:
                        1. Questions should be specific to the role and tech stack
                        2. Adjust complexity based on experience level
                        3. Include both technical and behavioral questions
                        4. Focus on required skills and project types
                        5. Make questions open-ended and discussion-friendly
                        6. Avoid yes/no questions
                        7. Include scenario-based questions for mid/senior levels

                        Return ONLY a JSON array of questions, no other text."""
                    },
                    {
                        "role": "user",
                        "content": f"Generate interview questions for this talent profile:\n{context}"
                    }
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            questions_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if questions_text.startswith('```json'):
                questions_text = questions_text[7:-3].strip()
            elif questions_text.startswith('```'):
                questions_text = questions_text[3:-3].strip()
            
            questions = json.loads(questions_text)
            
            # Ensure we have a list of strings
            if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                return questions
            else:
                return self._get_fallback_questions(talent_profile)
                
        except Exception as e:
            print(f"Error generating questions: {e}")
            return self._get_fallback_questions(talent_profile)

    def _get_fallback_questions(self, talent_profile: TalentProfile) -> List[str]:
        """Fallback questions when generation fails"""
        base_questions = [
            f"What interests you most about working in {talent_profile.industry}?",
            f"How do you stay updated with the latest trends in {talent_profile.role}?",
            f"Can you walk me through a challenging project you've worked on?",
            f"How do you approach problem-solving when you encounter a difficult technical issue?",
            f"What's your experience with {talent_profile.tech_stack[0] if talent_profile.tech_stack else 'your main technology'}?",
            f"How do you prioritize tasks when working on multiple projects?",
            f"Describe a time when you had to learn a new technology quickly.",
            f"What do you think are the most important qualities for a {talent_profile.role}?"
        ]
        return base_questions

    async def start_interview(self, candidate_id: str, talent_profile: TalentProfile) -> Dict[str, Any]:
        """Initialize a new interview session with dynamic questions"""
        session_id = f"session_{candidate_id}_{int(time.time())}"
        
        # Generate questions based on talent profile
        generated_questions = await self.generate_interview_questions(talent_profile)
        
        # Combine welcome message with generated questions
        all_questions = [self.welcome_message] + generated_questions
        
        session = InterviewSession(
            session_id=session_id,
            candidate_id=candidate_id,
            talent_profile=talent_profile,
            status="active",
            current_question=0,
            questions=all_questions,
            responses=[],
            overall_score=0.0,
            recommendations=[],
            total_cost=0.0,
            duration_minutes=0,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.sessions[session_id] = session
        
        return {
            "session_id": session_id,
            "status": "started",
            "first_question": all_questions[0],
            "estimated_duration": f"{len(all_questions) * 2}-{len(all_questions) * 3} minutes",
            "total_questions": len(all_questions),
            "role": talent_profile.role,
            "experience_level": talent_profile.experience_level,
            "tech_stack": talent_profile.tech_stack
        }

    # Replace the process_audio_response method in your AIInterviewEngine class with this fixed version:

    async def process_audio_response(self, session_id: str, audio_data: bytes) -> Dict[str, Any]:
        """Process candidate's audio response and generate next question - FIXED VERSION"""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        start_time = time.time()
        
        try:
            # Get current question context
            current_question = session.questions[session.current_question]
            evaluation_criteria = self._get_evaluation_criteria(session.talent_profile)
            
            # HYBRID APPROACH: Try audio-preview first, fallback to Whisper+GPT-4.1
            try:
                # METHOD 1: Try gpt-4o-audio-preview with text-only modality (more stable)
                audio_b64 = base64.b64encode(audio_data).decode()
                
                response = await self.client.chat.completions.create(
                    model="gpt-4o-audio-preview-2025-06-03",
                    messages=[
                        {
                            "role": "system",
                            "content": self._build_system_prompt(session.talent_profile, evaluation_criteria)
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Current question: {current_question}\n\nPlease analyze the candidate's audio response and provide structured feedback in JSON format."
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
                    modalities=["text"],  # ✅ FIXED: Only text output, no audio generation
                    temperature=0.7,
                    max_tokens=1000
                )
                
                # Parse structured response
                analysis = self._parse_ai_response(response.choices[0].message.content)
                audio_response = None  # No audio generation in this method
                cost = self._calculate_cost(response.usage)
                
                # Verify we got a good response
                if analysis.get("transcript") != "Response processing error" and len(analysis.get("transcript", "")) > 5:
                    print(f"✅ Primary audio processing successful")
                else:
                    raise Exception("Primary processing returned error, trying fallback")
                    
            except Exception as e:
                print(f"⚠️  Primary audio processing failed: {e}, trying fallback...")
                
                # METHOD 2: Fallback to Whisper + GPT-4.1 (more reliable)
                try:
                    # Step 1: Transcribe with Whisper
                    audio_file = io.BytesIO(audio_data)
                    audio_file.name = "audio.wav"
                    
                    transcription_response = await self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                    
                    transcript = transcription_response.text
                    print(f"✅ Whisper transcription: '{transcript[:50]}...'")
                    
                    # Step 2: Analyze with GPT-4.1
                    analysis_response = await self.client.chat.completions.create(
                        model="gpt-4.1-2025-04-14",
                        messages=[
                            {
                                "role": "system",
                                "content": f"""You are Anisa, an expert AI interviewer for {session.talent_profile.role} positions.

                                EVALUATION CRITERIA:
                                {self._format_criteria(evaluation_criteria)}

                                Current question: {current_question}

                                Analyze the candidate's response and return JSON:
                                {{
                                    "transcript": "exact transcript text",
                                    "score": float (0-10 scale),
                                    "key_points": ["point1", "point2", "point3"],
                                    "follow_ups": ["question1", "question2"],
                                    "sentiment": "positive/neutral/negative",
                                    "confidence": float (0-1),
                                    "tech_assessment": "brief assessment"
                                }}"""
                            },
                            {
                                "role": "user",
                                "content": f"Candidate transcript: '{transcript}'\n\nPlease evaluate this response."
                            }
                        ],
                        temperature=0.7,
                        max_tokens=800
                    )
                    
                    # Parse the analysis
                    analysis = self._parse_ai_response(analysis_response.choices[0].message.content)
                    analysis["transcript"] = transcript  # Ensure we use the Whisper transcript
                    
                    # Calculate combined cost
                    transcription_cost = 0.006  # Rough estimate for Whisper
                    analysis_cost = self._calculate_cost(analysis_response.usage, "gpt-4.1-2025-04-14")
                    cost = transcription_cost + analysis_cost
                    
                    audio_response = None
                    print(f"✅ Fallback processing successful")
                    
                except Exception as fallback_error:
                    print(f"❌ Fallback processing also failed: {fallback_error}")
                    return {
                        "error": f"All audio processing methods failed: {str(e)} | {str(fallback_error)}",
                        "session_id": session_id,
                        "retry_suggested": True
                    }
            
            # Update session cost
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
            if session.current_question < len(session.questions):
                next_question = session.questions[session.current_question]
            else:
                # Interview complete
                session.status = "completed"
                session.overall_score = self._calculate_overall_score(session)
                session.recommendations = await self._generate_recommendations(session)
                session.duration_minutes = int((time.time() - start_time) / 60)
            
            return {
                "session_id": session_id,
                "status": session.status,
                "response_analysis": asdict(interview_response),
                "next_question": next_question,
                "progress": {
                    "current": session.current_question,
                    "total": len(session.questions),
                    "percentage": (session.current_question / len(session.questions)) * 100
                },
                "cost_tracking": {
                    "current_cost": cost,
                    "total_cost": session.total_cost,
                    "estimated_final": session.total_cost * (len(session.questions) / max(session.current_question, 1))
                }
            }
            
        except Exception as e:
            return {
                "error": f"Processing failed: {str(e)}",
                "session_id": session_id,
                "retry_suggested": True
            }

    # Also add this helper method to your AIInterviewEngine class:
    def _build_system_prompt(self, talent_profile: TalentProfile, criteria: Dict[str, float]) -> str:
        """Build position-specific system prompt"""
        criteria_text = self._format_criteria(criteria)
        tech_stack_text = ", ".join(talent_profile.tech_stack)
    
        return f"""You are Anisa, an expert AI interviewer for {talent_profile.role} positions at Nouvells. 

                CANDIDATE PROFILE:
                - Role: {talent_profile.role}
                - Experience Level: {talent_profile.experience_level}
                - Tech Stack: {tech_stack_text}
                - Industry: {talent_profile.industry}
                - Years of Experience: {talent_profile.years_experience}

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
                    "areas_evaluated": {{"criteria": score}},
                    "tech_assessment": "brief assessment of technical mentions"
                }}

                INSTRUCTIONS:
                1. Transcribe the audio accurately
                2. Evaluate based on the criteria above and candidate's experience level
                3. Pay attention to mentions of required tech stack: {tech_stack_text}
                4. Identify 2-3 key strengths or concerns
                5. Suggest relevant follow-up questions
                6. Assess confidence level in your evaluation
                7. Keep responses professional and constructive
                8. Consider the candidate's experience level when scoring

                Return only the JSON response, no other text."""
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
        except Exception as e:
            print(f"JSON parsing error: {e}")
            print(f"Content: {content}")
            # Fallback parsing
            return {
                "transcript": "Response processing error",
                "score": 5.0,
                "key_points": ["Unable to parse response"],
                "follow_ups": ["Could you please repeat that?"],
                "sentiment": "neutral",
                "confidence": 0.5,
                "tech_assessment": "Unable to assess"
            }

    def _get_evaluation_criteria(self, talent_profile: TalentProfile) -> Dict[str, float]:
        """Get evaluation criteria based on experience level"""
        return self.evaluation_weights.get(talent_profile.experience_level, self.evaluation_weights["mid"])

    def _format_criteria(self, criteria: Dict[str, float]) -> str:
        """Format evaluation criteria for prompts"""
        return "\n".join([f"- {k.replace('_', ' ').title()}: {v*100}%" for k, v in criteria.items()])

    def _calculate_cost(self, usage, model_name: str = "gpt-4o-audio-preview-2025-06-03") -> float:
        """Calculate cost based on token usage and model"""
        model_costs = {
            "gpt-4o-audio-preview-2025-06-03": {"input": 0.005, "output": 0.015},
            "gpt-4.1-2025-04-14": {"input": 0.003, "output": 0.012},
            "whisper-1": {"input": 0.006, "output": 0}
        }
        
        if model_name not in model_costs:
            return 0.0
        
        costs = model_costs[model_name]
        input_cost = (usage.prompt_tokens / 1000) * costs["input"]
        output_cost = (usage.completion_tokens / 1000) * costs["output"]
        
        return input_cost + output_cost

    def _calculate_overall_score(self, session: InterviewSession) -> float:
        """Calculate weighted overall score"""
        if not session.responses:
            return 0.0
        
        # Calculate average score
        total_score = sum(resp.evaluation_score for resp in session.responses)
        return total_score / len(session.responses)

    async def _generate_recommendations(self, session: InterviewSession) -> List[str]:
        """Generate dynamic hiring recommendations based on talent profile"""
        try:
            context = {
                "role": session.talent_profile.role,
                "experience_level": session.talent_profile.experience_level,
                "overall_score": session.overall_score,
                "tech_stack": session.talent_profile.tech_stack,
                "responses_count": len(session.responses),
                "key_points": [point for resp in session.responses for point in resp.key_points]
            }
            
            response = await self.client.chat.completions.create(
                model="gpt-4.1-2025-04-14",
                messages=[
                    {
                        "role": "system",
                        "content": """Generate hiring recommendations based on interview performance. 
                        Return 3-5 specific recommendations as a JSON array of strings."""
                    },
                    {
                        "role": "user",
                        "content": f"Generate recommendations for: {json.dumps(context)}"
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            recommendations_text = response.choices[0].message.content.strip()
            if recommendations_text.startswith('```json'):
                recommendations_text = recommendations_text[7:-3].strip()
            
            recommendations = json.loads(recommendations_text)
            return recommendations if isinstance(recommendations, list) else []
            
        except Exception as e:
            # Fallback recommendations
            score = session.overall_score
            if score >= 8.0:
                return ["Strong hire - excellent candidate", "Proceed to final round", "Consider for senior role"]
            elif score >= 6.5:
                return ["Hire - good candidate", "Standard onboarding", "Monitor performance in first 90 days"]
            elif score >= 5.0:
                return ["Weak hire - proceed with caution", "Additional interview recommended", "Consider for junior role"]
            else:
                return ["No hire - significant concerns", "Skills gap too large", "Recommend alternative positions"]


    async def get_interview_summary(self, session_id: str) -> Dict[str, Any]:
        """Get complete interview summary and evaluation"""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        
        # Generate comprehensive evaluation
        evaluation_insights = await self._generate_final_evaluation(session)
        
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
                "role": session.talent_profile.role,
                "experience_level": session.talent_profile.experience_level,
                "tech_stack": session.talent_profile.tech_stack,
                "score": session.overall_score,
                "duration": session.duration_minutes,
                "responses": len(session.responses)
            }
        }

    async def _generate_final_evaluation(self, session: InterviewSession) -> Dict[str, Any]:
        """Generate comprehensive final evaluation"""
        try:
            context = {
                "talent_profile": asdict(session.talent_profile),
                "responses": [{"transcript": resp.transcript, "score": resp.evaluation_score, "key_points": resp.key_points} for resp in session.responses],
                "overall_score": session.overall_score
            }
            
            response = await self.client.chat.completions.create(
                model="gpt-4.1-2025-04-14",  #
                messages=[
                    {
                        "role": "system",
                        "content": """Generate a comprehensive evaluation summary. Return JSON with:
                        {
                            "strengths": ["strength1", "strength2"],
                            "areas_for_improvement": ["area1", "area2"],
                            "cultural_fit": "assessment",
                            "technical_competency": "assessment",
                            "risk_factors": ["risk1", "risk2"],
                            "role_match_percentage": number
                        }"""
                    },
                    {
                        "role": "user",
                        "content": f"Evaluate this interview: {json.dumps(context)}"
                    }
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            eval_text = response.choices[0].message.content.strip()
            if eval_text.startswith('```json'):
                eval_text = eval_text[7:-3].strip()
            
            return json.loads(eval_text)
            
        except Exception as e:
            return {
                "strengths": ["Communication skills", "Technical knowledge"],
                "areas_for_improvement": ["System design", "Leadership experience"],
                "cultural_fit": "Good potential match",
                "technical_competency": "Meets requirements",
                "risk_factors": ["Limited experience in some areas"],
                "role_match_percentage": int(session.overall_score * 10)
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
            "role": session.talent_profile.role,
            "experience_level": session.talent_profile.experience_level,
            "reasoning": session.recommendations[0] if session.recommendations else "Standard evaluation"
        }
# API endpoints for your web developers

class InterviewAPI:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.engine = AIInterviewEngine(api_key=api_key)
    
    async def start_interview_endpoint(self, candidate_id: str, talent_profile_data: Dict[str, Any]):
        """POST /api/interview/start"""
        talent_profile = TalentProfile(**talent_profile_data)
        return await self.engine.start_interview(candidate_id, talent_profile)
    
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
                "total_questions": len(session.questions),
                "role": session.talent_profile.role,
                "total_cost": session.total_cost
            }
        return {"error": "Session not found"}

# Example usage
async def test_dynamic_interview():
    """Test the dynamic interview pipeline"""
    
    # Initialize the engine
    engine = AIInterviewEngine(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create a sample talent profile
    talent_profile = TalentProfile(
        role="Full Stack Developer",
        experience_level="mid",
        tech_stack=["React", "Node.js", "Python", "PostgreSQL", "AWS"],
        industry="fintech",
        required_skills=["JavaScript", "REST APIs", "Database Design", "Git"],
        soft_skills=["teamwork", "communication", "problem-solving"],
        years_experience=4,
        education_level="Bachelor's",
        certifications=["AWS Certified Developer"],
        project_types=["web applications", "mobile apps", "APIs"]
    )
    
    # Start interview
    start_result = await engine.start_interview("candidate_456", talent_profile)
    print("Interview Started:", json.dumps(start_result, indent=2))
    
    # The rest would work the same as before...

if __name__ == "__main__":
    # Run test
    asyncio.run(test_dynamic_interview())