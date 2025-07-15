#!/usr/bin/env python3
"""
FastAPI Server for AI Interview System
Complete implementation with dynamic role support and talent profiles
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
import uvicorn
import os
import base64
import json
import asyncio
from datetime import datetime
import logging

# Import your AI engine (assuming it's in the same directory)
from ai_interview_engine import InterviewAPI, AIInterviewEngine, TalentProfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Nouvells AI Interview System API",
    description="Dynamic REST API for conducting AI-powered candidate interviews with any role",
    version="2.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc"  # ReDoc at /redoc
)

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the interview API
interview_api = InterviewAPI()

# Pydantic models for request/response validation
class TalentProfileRequest(BaseModel):
    """Talent profile for dynamic interview generation"""
    role: str = Field(..., description="Job role (e.g., 'Full Stack Developer', 'DevOps Engineer')")
    experience_level: str = Field(..., description="Experience level", 
                                 pattern="^(entry|mid|senior|lead)$")
    tech_stack: List[str] = Field(..., description="List of technologies", 
                                 example=["Python", "React", "AWS"])
    industry: str = Field(..., description="Industry domain", 
                         example="fintech")
    required_skills: List[str] = Field(..., description="Required skills for the role")
    soft_skills: List[str] = Field(default_factory=list, 
                                  description="Soft skills to evaluate")
    years_experience: int = Field(..., ge=0, le=50, 
                                 description="Years of professional experience")
    education_level: str = Field(default="Bachelor's", 
                               description="Education level")
    certifications: List[str] = Field(default_factory=list, 
                                    description="Professional certifications")
    project_types: List[str] = Field(..., description="Types of projects worked on")

    @validator('experience_level')
    def validate_experience_level(cls, v):
        valid_levels = ['entry', 'mid', 'senior', 'lead']
        if v.lower() not in valid_levels:
            raise ValueError(f'Experience level must be one of: {valid_levels}')
        return v.lower()

class StartInterviewRequest(BaseModel):
    candidate_id: str = Field(..., description="Unique identifier for the candidate")
    talent_profile: TalentProfileRequest = Field(..., description="Talent profile for dynamic interview")
    candidate_name: Optional[str] = Field(None, description="Candidate's full name")
    recruiter_id: Optional[str] = Field(None, description="ID of the recruiter")
    interview_type: Optional[str] = Field("screening", description="Type of interview")
    company_info: Optional[Dict[str, Any]] = Field(None, description="Company-specific information")

class SubmitResponseRequest(BaseModel):
    session_id: str = Field(..., description="Interview session identifier")
    audio_base64: str = Field(..., description="Base64 encoded audio data")
    question_id: Optional[int] = Field(None, description="Current question ID")

class InterviewStatusResponse(BaseModel):
    session_id: str
    status: str
    progress: Dict[str, Any]
    total_cost: float
    current_question: Optional[str] = None
    role: str
    experience_level: str

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    timestamp: str
    session_id: Optional[str] = None

class BulkInterviewRequest(BaseModel):
    """For starting multiple interviews at once"""
    interviews: List[StartInterviewRequest] = Field(..., description="List of interview requests")
    batch_id: Optional[str] = Field(None, description="Batch identifier")

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "details": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "features": {
            "dynamic_roles": True,
            "talent_profiles": True,
            "ai_question_generation": True
        },
        "services": {
            "openai": "connected",
            "ai_engine": "ready"
        }
    }

# Interview Management Endpoints

@app.post("/api/interview/start")
async def start_interview(request: StartInterviewRequest):
    """
    Start a new interview session with dynamic role support
    
    - **candidate_id**: Unique identifier for the candidate
    - **talent_profile**: Complete talent profile including role, tech stack, experience level
    - **candidate_name**: Optional candidate name for personalization
    - **recruiter_id**: Optional recruiter identifier
    - **interview_type**: Type of interview (screening, technical, behavioral)
    """
    try:
        logger.info(f"Starting interview for candidate {request.candidate_id}, role: {request.talent_profile.role}")
        
        # Convert request to TalentProfile dataclass
        talent_profile = TalentProfile(
            role=request.talent_profile.role,
            experience_level=request.talent_profile.experience_level,
            tech_stack=request.talent_profile.tech_stack,
            industry=request.talent_profile.industry,
            required_skills=request.talent_profile.required_skills,
            soft_skills=request.talent_profile.soft_skills,
            years_experience=request.talent_profile.years_experience,
            education_level=request.talent_profile.education_level,
            certifications=request.talent_profile.certifications,
            project_types=request.talent_profile.project_types
        )
        
        result = await interview_api.start_interview_endpoint(
            request.candidate_id, 
            talent_profile.__dict__
        )
        
        # Add additional metadata
        result.update({
            "candidate_name": request.candidate_name,
            "recruiter_id": request.recruiter_id,
            "interview_type": request.interview_type,
            "company_info": request.company_info,
            "created_at": datetime.now().isoformat()
        })
        
        logger.info(f"Interview started successfully: {result['session_id']}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to start interview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start interview: {str(e)}")

@app.post("/api/interview/start-bulk")
async def start_bulk_interviews(request: BulkInterviewRequest, background_tasks: BackgroundTasks):
    """
    Start multiple interview sessions at once
    
    - **interviews**: List of interview requests
    - **batch_id**: Optional batch identifier for tracking
    """
    try:
        results = []
        errors = []
        
        for i, interview_req in enumerate(request.interviews):
            try:
                result = await start_interview(interview_req)
                results.append({
                    "index": i,
                    "candidate_id": interview_req.candidate_id,
                    "session_id": result["session_id"],
                    "status": "started"
                })
            except Exception as e:
                errors.append({
                    "index": i,
                    "candidate_id": interview_req.candidate_id,
                    "error": str(e)
                })
        
        return {
            "batch_id": request.batch_id,
            "total_requested": len(request.interviews),
            "successful": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk interview creation failed: {str(e)}")

@app.post("/api/interview/response")
async def submit_response(request: SubmitResponseRequest):
    """
    Submit candidate's audio response to current question
    
    - **session_id**: Active interview session ID
    - **audio_base64**: Base64 encoded audio data (WAV format recommended)
    - **question_id**: Optional current question ID for validation
    """
    try:
        logger.info(f"Processing response for session {request.session_id}")
        
        # Validate session exists
        session_status = await interview_api.get_session_status_endpoint(request.session_id)
        if "error" in session_status:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session_status["status"] != "active":
            raise HTTPException(status_code=400, detail="Session is not active")
        
        # Validate audio data
        try:
            audio_data = base64.b64decode(request.audio_base64)
            if len(audio_data) == 0:
                raise ValueError("Empty audio data")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio data: {str(e)}")
        
        result = await interview_api.submit_response_endpoint(
            request.session_id, 
            request.audio_base64
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        logger.info(f"Response processed successfully for session {request.session_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process response: {str(e)}")

@app.get("/api/interview/summary/{session_id}")
async def get_summary(session_id: str):
    """
    Get complete interview summary and evaluation
    
    - **session_id**: Interview session ID
    
    Returns detailed analysis, scores, and hiring recommendations
    """
    try:
        logger.info(f"Generating summary for session {session_id}")
        
        result = await interview_api.get_summary_endpoint(session_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        logger.info(f"Summary generated for session {session_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

@app.get("/api/interview/status/{session_id}")
async def get_status(session_id: str):
    """
    Get current interview session status
    
    - **session_id**: Interview session ID
    
    Returns current progress, cost, and session state
    """
    try:
        result = await interview_api.get_session_status_endpoint(session_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

# Role and Profile Management Endpoints

@app.get("/api/roles/suggestions")
async def get_role_suggestions(query: Optional[str] = None):
    """
    Get role suggestions for autocomplete
    
    - **query**: Optional search query to filter roles
    """
    try:
        # Common role suggestions - could be enhanced with a database
        all_roles = [
            "Full Stack Developer", "Frontend Developer", "Backend Developer",
            "DevOps Engineer", "Data Scientist", "Data Engineer", "Data Analyst",
            "Product Manager", "Project Manager", "UI/UX Designer",
            "Mobile Developer", "iOS Developer", "Android Developer",
            "Machine Learning Engineer", "AI Engineer", "Software Architect",
            "QA Engineer", "Test Automation Engineer", "Site Reliability Engineer",
            "Cybersecurity Analyst", "Network Engineer", "Database Administrator",
            "Business Analyst", "Technical Writer", "Scrum Master"
        ]
        
        if query:
            filtered_roles = [role for role in all_roles if query.lower() in role.lower()]
            return {"roles": filtered_roles[:10]}
        
        return {"roles": all_roles}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get role suggestions: {str(e)}")

@app.get("/api/tech-stack/suggestions")
async def get_tech_stack_suggestions(role: Optional[str] = None):
    """
    Get technology stack suggestions based on role
    
    - **role**: Optional role to get relevant tech stack
    """
    try:
        tech_suggestions = {
            "default": ["JavaScript", "Python", "Java", "C++", "SQL", "Git", "Linux"],
            "frontend": ["React", "Vue.js", "Angular", "HTML", "CSS", "TypeScript", "Webpack"],
            "backend": ["Node.js", "Django", "FastAPI", "Spring Boot", "Express.js", "PostgreSQL"],
            "mobile": ["React Native", "Flutter", "Swift", "Kotlin", "Xamarin"],
            "data": ["Python", "R", "SQL", "Pandas", "NumPy", "TensorFlow", "PyTorch", "Spark"],
            "devops": ["Docker", "Kubernetes", "Jenkins", "Terraform", "AWS", "Azure", "GCP"],
            "cloud": ["AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Serverless"]
        }
        
        if role:
            role_lower = role.lower()
            if "frontend" in role_lower or "ui" in role_lower:
                return {"technologies": tech_suggestions["frontend"]}
            elif "backend" in role_lower:
                return {"technologies": tech_suggestions["backend"]}
            elif "mobile" in role_lower:
                return {"technologies": tech_suggestions["mobile"]}
            elif "data" in role_lower or "ml" in role_lower or "ai" in role_lower:
                return {"technologies": tech_suggestions["data"]}
            elif "devops" in role_lower or "sre" in role_lower:
                return {"technologies": tech_suggestions["devops"]}
        
        return {"technologies": tech_suggestions["default"]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tech stack suggestions: {str(e)}")

@app.post("/api/profile/validate")
async def validate_talent_profile(profile: TalentProfileRequest):
    """
    Validate a talent profile before starting an interview
    
    - **profile**: Talent profile to validate
    """
    try:
        # Perform validation logic
        issues = []
        
        # Check required fields
        if not profile.role.strip():
            issues.append("Role is required")
        
        if not profile.tech_stack:
            issues.append("At least one technology is required")
        
        if profile.years_experience < 0:
            issues.append("Years of experience cannot be negative")
        
        # Experience level validation
        exp_mapping = {"entry": (0, 2), "mid": (2, 7), "senior": (7, 12), "lead": (10, 50)}
        if profile.experience_level in exp_mapping:
            min_exp, max_exp = exp_mapping[profile.experience_level]
            if not (min_exp <= profile.years_experience <= max_exp):
                issues.append(f"Years of experience ({profile.years_experience}) doesn't match experience level ({profile.experience_level})")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "suggestions": {
                "estimated_interview_duration": f"{len(profile.required_skills) * 2}-{len(profile.required_skills) * 3} minutes",
                "question_count": min(8, max(5, len(profile.required_skills) + 2))
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile validation failed: {str(e)}")

# Additional utility endpoints

@app.post("/api/interview/pause/{session_id}")
async def pause_interview(session_id: str):
    """Pause an active interview session"""
    try:
        # Get current session
        session = interview_api.engine.sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.status != "active":
            raise HTTPException(status_code=400, detail="Session is not active")
        
        session.status = "paused"
        
        return {
            "session_id": session_id,
            "status": "paused",
            "message": "Interview paused successfully",
            "current_progress": {
                "question": session.current_question,
                "total": len(session.questions)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to pause interview: {str(e)}")

@app.post("/api/interview/resume/{session_id}")
async def resume_interview(session_id: str):
    """Resume a paused interview session"""
    try:
        session = interview_api.engine.sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.status != "paused":
            raise HTTPException(status_code=400, detail="Session is not paused")
        
        session.status = "active"
        
        # Get current question
        current_question = None
        if session.current_question < len(session.questions):
            current_question = session.questions[session.current_question]
        
        return {
            "session_id": session_id,
            "status": "active",
            "current_question": current_question,
            "message": "Interview resumed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resume interview: {str(e)}")

@app.delete("/api/interview/{session_id}")
async def cancel_interview(session_id: str):
    """Cancel and delete an interview session"""
    try:
        if session_id in interview_api.engine.sessions:
            session = interview_api.engine.sessions[session_id]
            del interview_api.engine.sessions[session_id]
            return {
                "session_id": session_id,
                "status": "cancelled",
                "message": "Interview cancelled successfully",
                "role": session.talent_profile.role,
                "candidate_id": session.candidate_id
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel interview: {str(e)}")

@app.get("/api/interview/sessions")
async def list_sessions(
    status: Optional[str] = None,
    role: Optional[str] = None,
    experience_level: Optional[str] = None
):
    """
    List interview sessions with optional filtering
    
    - **status**: Filter by session status (active, completed, paused)
    - **role**: Filter by role
    - **experience_level**: Filter by experience level
    """
    try:
        sessions = []
        for session_id, session in interview_api.engine.sessions.items():
            # Apply filters
            if status and session.status != status:
                continue
            if role and session.talent_profile.role.lower() != role.lower():
                continue
            if experience_level and session.talent_profile.experience_level != experience_level:
                continue
            
            sessions.append({
                "session_id": session_id,
                "candidate_id": session.candidate_id,
                "role": session.talent_profile.role,
                "experience_level": session.talent_profile.experience_level,
                "tech_stack": session.talent_profile.tech_stack,
                "status": session.status,
                "progress": session.current_question,
                "total_questions": len(session.questions),
                "overall_score": session.overall_score,
                "created_at": session.created_at
            })
        
        return {
            "total_sessions": len(sessions),
            "filters_applied": {
                "status": status,
                "role": role,
                "experience_level": experience_level
            },
            "sessions": sessions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

# File upload endpoint for audio
@app.post("/api/interview/upload-audio/{session_id}")
async def upload_audio_file(session_id: str, audio_file: UploadFile = File(...)):
    """
    Upload audio file directly instead of base64
    
    - **session_id**: Interview session ID
    - **audio_file**: Audio file (WAV, MP3, etc.)
    """
    try:
        # Validate file type
        allowed_types = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg", "audio/webm"]
        if audio_file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: {allowed_types}"
            )
        
        # Check file size (max 10MB)
        content = await audio_file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
        
        # Convert to base64
        audio_base64 = base64.b64encode(content).decode()
        
        # Process the response
        result = await interview_api.submit_response_endpoint(session_id, audio_base64)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process audio file: {str(e)}")

# Analytics and monitoring endpoints

@app.get("/api/analytics/cost-summary")
async def get_cost_summary():
    """Get cost analytics across all sessions"""
    try:
        total_cost = 0
        total_sessions = 0
        active_sessions = 0
        by_role = {}
        by_experience = {}
        
        for session in interview_api.engine.sessions.values():
            total_cost += session.total_cost
            total_sessions += 1
            if session.status == "active":
                active_sessions += 1
            
            # Role breakdown
            role = session.talent_profile.role
            if role not in by_role:
                by_role[role] = {"count": 0, "total_cost": 0}
            by_role[role]["count"] += 1
            by_role[role]["total_cost"] += session.total_cost
            
            # Experience breakdown
            exp = session.talent_profile.experience_level
            if exp not in by_experience:
                by_experience[exp] = {"count": 0, "total_cost": 0}
            by_experience[exp]["count"] += 1
            by_experience[exp]["total_cost"] += session.total_cost
        
        avg_cost_per_session = total_cost / total_sessions if total_sessions > 0 else 0
        
        return {
            "total_cost": round(total_cost, 4),
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "average_cost_per_session": round(avg_cost_per_session, 4),
            "breakdown_by_role": by_role,
            "breakdown_by_experience": by_experience,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cost summary: {str(e)}")

@app.get("/api/analytics/performance")
async def get_performance_metrics():
    """Get system performance metrics"""
    try:
        sessions = interview_api.engine.sessions
        
        # Calculate metrics
        completed_sessions = [s for s in sessions.values() if s.status == "completed"]
        avg_score = sum(s.overall_score for s in completed_sessions) / len(completed_sessions) if completed_sessions else 0
        avg_duration = sum(s.duration_minutes for s in completed_sessions) / len(completed_sessions) if completed_sessions else 0
        
        # Role and experience breakdown
        role_stats = {}
        experience_stats = {}
        tech_usage = {}
        
        for session in completed_sessions:
            role = session.talent_profile.role
            exp = session.talent_profile.experience_level
            
            # Role stats
            if role not in role_stats:
                role_stats[role] = {"count": 0, "avg_score": 0, "scores": []}
            role_stats[role]["count"] += 1
            role_stats[role]["scores"].append(session.overall_score)
            
            # Experience stats
            if exp not in experience_stats:
                experience_stats[exp] = {"count": 0, "avg_score": 0, "scores": []}
            experience_stats[exp]["count"] += 1
            experience_stats[exp]["scores"].append(session.overall_score)
            
            # Tech usage
            for tech in session.talent_profile.tech_stack:
                tech_usage[tech] = tech_usage.get(tech, 0) + 1
        
        # Calculate averages
        for role_data in role_stats.values():
            role_data["avg_score"] = sum(role_data["scores"]) / len(role_data["scores"])
            del role_data["scores"]
        
        for exp_data in experience_stats.values():
            exp_data["avg_score"] = sum(exp_data["scores"]) / len(exp_data["scores"])
            del exp_data["scores"]
        
        return {
            "total_interviews": len(sessions),
            "completed_interviews": len(completed_sessions),
            "average_score": round(avg_score, 2),
            "average_duration_minutes": round(avg_duration, 1),
            "role_performance": role_stats,
            "experience_performance": experience_stats,
            "popular_technologies": dict(sorted(tech_usage.items(), key=lambda x: x[1], reverse=True)[:10]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

# Testing and development endpoints

@app.post("/api/test/generate-sample-interview")
async def generate_sample_interview(role: str = "Full Stack Developer"):
    """Generate a sample interview for testing purposes"""
    try:
        # Create test talent profile
        sample_profile = TalentProfileRequest(
            role=role,
            experience_level="mid",
            tech_stack=["Python", "React", "PostgreSQL", "Docker"],
            industry="technology",
            required_skills=["JavaScript", "Python", "SQL", "Git"],
            soft_skills=["communication", "teamwork", "problem-solving"],
            years_experience=4,
            education_level="Bachelor's",
            certifications=["AWS Certified Developer"],
            project_types=["web applications", "APIs", "microservices"]
        )
        
        # Create test interview request
        interview_request = StartInterviewRequest(
            candidate_id=f"test_candidate_{int(datetime.now().timestamp())}",
            talent_profile=sample_profile,
            candidate_name="Test Candidate",
            interview_type="sample"
        )
        
        # Start interview
        result = await start_interview(interview_request)
        session_id = result["session_id"]

        # Simulate responses with dummy audio
        try:
            # Check if test audio file exists
            audio_file_path = "output_5.wav"
            if os.path.exists(audio_file_path):
                with open(audio_file_path, "rb") as audio_file:
                    dummy_audio = base64.b64encode(audio_file.read()).decode()
                
                # Submit a sample response
                await interview_api.submit_response_endpoint(session_id, dummy_audio)
            else:
                logger.warning("Test audio file not found, skipping response simulation")
        except Exception as e:
            logger.warning(f"Could not simulate audio response: {e}")

        # Get summary
        try:
            summary = await interview_api.get_summary_endpoint(session_id)
        except:
            summary = {"note": "Summary not available - interview may not be complete"}

        return {
            "message": "Sample interview generated successfully",
            "session_id": session_id,
            "role": role,
            "sample_questions_generated": len(result.get("total_questions", 0)),
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate sample interview: {str(e)}")

@app.post("/api/interview/questions/preview")
async def preview_questions(talent_profile: TalentProfileRequest):
    """
    Preview questions that would be generated for a talent profile
    
    - **talent_profile**: Talent profile to generate questions for
    """
    try:
        # Convert to TalentProfile dataclass
        profile = TalentProfile(
            role=talent_profile.role,
            experience_level=talent_profile.experience_level,
            tech_stack=talent_profile.tech_stack,
            industry=talent_profile.industry,
            required_skills=talent_profile.required_skills,
            soft_skills=talent_profile.soft_skills,
            years_experience=talent_profile.years_experience,
            education_level=talent_profile.education_level,
            certifications=talent_profile.certifications,
            project_types=talent_profile.project_types
        )
        
        # Generate questions without starting an interview
        questions = await interview_api.engine.generate_interview_questions(profile)
        
        # Add the welcome message
        all_questions = [interview_api.engine.welcome_message] + questions
        
        return {
            "role": talent_profile.role,
            "experience_level": talent_profile.experience_level,
            "total_questions": len(all_questions),
            "estimated_duration": f"{len(all_questions) * 2}-{len(all_questions) * 3} minutes",
            "questions": all_questions,
            "evaluation_criteria": interview_api.engine._get_evaluation_criteria(profile)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preview questions: {str(e)}")

@app.get("/api/interview/export/{session_id}")
async def export_interview_data(session_id: str, format: str = "json"):
    """
    Export interview data in various formats
    
    - **session_id**: Interview session ID
    - **format**: Export format (json, csv, pdf)
    """
    try:
        if session_id not in interview_api.engine.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = interview_api.engine.sessions[session_id]
        
        if format.lower() == "json":
            export_data = {
                "session_info": {
                    "session_id": session_id,
                    "candidate_id": session.candidate_id,
                    "role": session.talent_profile.role,
                    "experience_level": session.talent_profile.experience_level,
                    "tech_stack": session.talent_profile.tech_stack,
                    "industry": session.talent_profile.industry,
                    "required_skills": session.talent_profile.required_skills,
                    "years_experience": session.talent_profile.years_experience,
                    "created_at": session.created_at,
                    "status": session.status,
                    "overall_score": session.overall_score,
                    "duration_minutes": session.duration_minutes
                },
                "questions_and_responses": [
                    {
                        "question_id": i,
                        "question": session.questions[resp.question_id] if resp.question_id < len(session.questions) else "Unknown",
                        "transcript": resp.transcript,
                        "score": resp.evaluation_score,
                        "key_points": resp.key_points,
                        "sentiment": resp.sentiment,
                        "confidence": resp.confidence,
                        "timestamp": resp.timestamp
                    }
                    for i, resp in enumerate(session.responses)
                ],
                "recommendations": session.recommendations,
                "cost_summary": {
                    "total_cost": session.total_cost,
                    "duration_minutes": session.duration_minutes,
                    "cost_per_response": session.total_cost / max(len(session.responses), 1)
                },
                "evaluation_criteria": interview_api.engine._get_evaluation_criteria(session.talent_profile),
                "export_metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "format": format,
                    "api_version": "2.0.0"
                }
            }
            
            return export_data
        
        else:
            raise HTTPException(status_code=400, detail="Only JSON format is currently supported")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/api/system/stats")
async def get_system_statistics():
    """Get comprehensive system statistics"""
    try:
        sessions = interview_api.engine.sessions
        
        # Basic stats
        total_sessions = len(sessions)
        active_sessions = len([s for s in sessions.values() if s.status == "active"])
        completed_sessions = len([s for s in sessions.values() if s.status == "completed"])
        paused_sessions = len([s for s in sessions.values() if s.status == "paused"])
        
        # Usage statistics
        unique_roles = len(set(s.talent_profile.role for s in sessions.values()))
        total_questions_asked = sum(len(s.questions) for s in sessions.values())
        total_responses = sum(len(s.responses) for s in sessions.values())
        
        # Performance metrics
        completed = [s for s in sessions.values() if s.status == "completed"]
        avg_score = sum(s.overall_score for s in completed) / len(completed) if completed else 0
        avg_duration = sum(s.duration_minutes for s in completed) / len(completed) if completed else 0
        total_cost = sum(s.total_cost for s in sessions.values())
        
        # Technology trends
        tech_frequency = {}
        for session in sessions.values():
            for tech in session.talent_profile.tech_stack:
                tech_frequency[tech] = tech_frequency.get(tech, 0) + 1
        
        top_technologies = dict(sorted(tech_frequency.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Role breakdown
        role_breakdown = {}
        experience_breakdown = {}
        industry_breakdown = {}
        
        for session in sessions.values():
            # Role stats
            role = session.talent_profile.role
            if role not in role_breakdown:
                role_breakdown[role] = {"count": 0, "avg_score": 0, "scores": []}
            role_breakdown[role]["count"] += 1
            if session.status == "completed":
                role_breakdown[role]["scores"].append(session.overall_score)
            
            # Experience stats
            exp = session.talent_profile.experience_level
            if exp not in experience_breakdown:
                experience_breakdown[exp] = {"count": 0, "avg_score": 0, "scores": []}
            experience_breakdown[exp]["count"] += 1
            if session.status == "completed":
                experience_breakdown[exp]["scores"].append(session.overall_score)
            
            # Industry stats
            industry = session.talent_profile.industry
            if industry not in industry_breakdown:
                industry_breakdown[industry] = {"count": 0}
            industry_breakdown[industry]["count"] += 1
        
        # Calculate averages and clean up
        for role_data in role_breakdown.values():
            if role_data["scores"]:
                role_data["avg_score"] = round(sum(role_data["scores"]) / len(role_data["scores"]), 2)
            del role_data["scores"]
        
        for exp_data in experience_breakdown.values():
            if exp_data["scores"]:
                exp_data["avg_score"] = round(sum(exp_data["scores"]) / len(exp_data["scores"]), 2)
            del exp_data["scores"]
        
        return {
            "session_statistics": {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "completed_sessions": completed_sessions,
                "paused_sessions": paused_sessions,
                "completion_rate": round(completed_sessions / total_sessions * 100, 2) if total_sessions > 0 else 0
            },
            "usage_statistics": {
                "unique_roles_interviewed": unique_roles,
                "total_questions_generated": total_questions_asked,
                "total_responses_processed": total_responses,
                "avg_questions_per_interview": round(total_questions_asked / total_sessions, 1) if total_sessions > 0 else 0
            },
            "performance_metrics": {
                "average_interview_score": round(avg_score, 2),
                "average_duration_minutes": round(avg_duration, 1),
                "total_ai_cost": round(total_cost, 4),
                "average_cost_per_interview": round(total_cost / total_sessions, 4) if total_sessions > 0 else 0
            },
            "breakdowns": {
                "by_role": role_breakdown,
                "by_experience_level": experience_breakdown,
                "by_industry": industry_breakdown
            },
            "technology_trends": {
                "most_popular_technologies": top_technologies,
                "total_unique_technologies": len(tech_frequency)
            },
            "system_info": {
                "api_version": "2.0.0",
                "features_enabled": ["dynamic_roles", "ai_question_generation", "talent_profiles"],
                "timestamp": datetime.now().isoformat(),
                "uptime_sessions": total_sessions
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system statistics: {str(e)}")
    


# Server startup and configuration

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Dynamic AI Interview System API...")
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set!")
        raise RuntimeError("OPENAI_API_KEY is required")
    
    # Test AI engine initialization
    try:
        test_profile = TalentProfile(
            role="Test Role",
            experience_level="mid",
            tech_stack=["Python"],
            industry="technology",
            required_skills=["Python"],
            soft_skills=["communication"],
            years_experience=3,
            education_level="Bachelor's",
            certifications=[],
            project_types=["web apps"]
        )
        logger.info("AI engine initialization test passed")
    except Exception as e:
        logger.error(f"AI engine initialization failed: {e}")
        raise RuntimeError("AI engine initialization failed")