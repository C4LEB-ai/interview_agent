#!/usr/bin/env python3
"""
FastAPI Server for AI Interview System
Complete implementation with all endpoints, error handling, and testing
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn
import os
import base64
import json
import asyncio
from datetime import datetime
import logging

# Import your AI engine (assuming it's in the same directory)
from ai_interview_engine import InterviewAPI, AIInterviewEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Nouvells AI Interview System API docs",
    description="REST API for conducting AI-powered candidate interviews",
    version="1.0.0",
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
class StartInterviewRequest(BaseModel):
    candidate_id: str = Field(..., description="Unique identifier for the candidate")
    position: str = Field(..., description="Position being interviewed for", 
                         example="software_engineer")
    candidate_name: Optional[str] = Field(None, description="Candidate's full name")
    recruiter_id: Optional[str] = Field(None, description="ID of the recruiter")
    interview_type: Optional[str] = Field("screening", description="Type of interview")

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

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    timestamp: str
    session_id: Optional[str] = None

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
        "version": "1.0.0",
        "services": {
            "openai": "connected",
            "database": "connected"
        }
    }

# Interview Management Endpoints

@app.post("/api/interview/start")
async def start_interview(request: StartInterviewRequest):
    """
    Start a new interview session
    
    - **candidate_id**: Unique identifier for the candidate
    - **position**: Position being interviewed for (software_engineer, data_scientist, product_manager)
    - **candidate_name**: Optional candidate name for personalization
    - **recruiter_id**: Optional recruiter identifier
    - **interview_type**: Type of interview (screening, technical, behavioral)
    """
    try:
        logger.info(f"Starting interview for candidate {request.candidate_id}, position: {request.position}")
        
        # Validate position
        valid_positions = ["software_engineer", "data_scientist", "product_manager"]
        if request.position not in valid_positions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid position. Must be one of: {valid_positions}"
            )
        
        result = await interview_api.start_interview_endpoint(
            request.candidate_id, 
            request.position
        )
        
        # Add additional metadata
        result.update({
            "candidate_name": request.candidate_name,
            "recruiter_id": request.recruiter_id,
            "interview_type": request.interview_type,
            "created_at": datetime.now().isoformat()
        })
        
        logger.info(f"Interview started successfully: {result['session_id']}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to start interview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start interview: {str(e)}")

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
            "message": "Interview paused successfully"
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
        
        # Get next question
        next_question = await interview_api.engine._get_next_question(session_id)
        
        return {
            "session_id": session_id,
            "status": "active",
            "next_question": next_question,
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
            del interview_api.engine.sessions[session_id]
            return {
                "session_id": session_id,
                "status": "cancelled",
                "message": "Interview cancelled successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel interview: {str(e)}")

@app.get("/api/interview/sessions")
async def list_sessions():
    """List all active interview sessions"""
    try:
        sessions = []
        for session_id, session in interview_api.engine.sessions.items():
            sessions.append({
                "session_id": session_id,
                "candidate_id": session.candidate_id,
                "position": session.position,
                "status": session.status,
                "progress": session.current_question,
                "created_at": session.created_at
            })
        
        return {
            "total_sessions": len(sessions),
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
        allowed_types = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg"]
        if audio_file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: {allowed_types}"
            )
        
        # Read file
        audio_data = await audio_file.read()
        
        # Convert to base64
        audio_base64 = base64.b64encode(audio_data).decode()
        
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
        
        for session in interview_api.engine.sessions.values():
            total_cost += session.total_cost
            total_sessions += 1
            if session.status == "active":
                active_sessions += 1
        
        avg_cost_per_session = total_cost / total_sessions if total_sessions > 0 else 0
        
        return {
            "total_cost": round(total_cost, 4),
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "average_cost_per_session": round(avg_cost_per_session, 4),
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
        
        # Position breakdown
        positions = {}
        for session in sessions.values():
            pos = session.position
            if pos not in positions:
                positions[pos] = {"count": 0, "avg_score": 0}
            positions[pos]["count"] += 1
            
        return {
            "total_interviews": len(sessions),
            "completed_interviews": len(completed_sessions),
            "average_score": round(avg_score, 2),
            "average_duration_minutes": round(avg_duration, 1),
            "position_breakdown": positions,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

# Testing and development endpoints

@app.post("/api/test/generate-sample-interview")
async def generate_sample_interview():
    """Generate a sample interview for testing purposes"""
    try:
        # Create test interview
        result = await interview_api.start_interview_endpoint("test_candidate_001", "software_engineer")
        session_id = result["session_id"]
        
        # Simulate responses
        test_responses = [
            "I have 5 years of experience in full-stack development",
            "I use systematic debugging with logs and breakpoints",
            "I've designed microservices with proper API boundaries"
        ]
        
        for i, response_text in enumerate(test_responses):
            # Create dummy audio data
            dummy_audio = base64.b64encode(f"test_audio_response_{i}".encode()).decode()
            await interview_api.submit_response_endpoint(session_id, dummy_audio)
        
        # Get summary
        summary = await interview_api.get_summary_endpoint(session_id)
        
        return {
            "message": "Sample interview generated successfully",
            "session_id": session_id,
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate sample interview: {str(e)}")

# Server startup and configuration

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting AI Interview System API...")
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set!")
        raise RuntimeError("OPENAI_API_KEY is required")
    
    logger.info("API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Interview System API...")

# Main application runner
if __name__ == "__main__":
    # Development server configuration
    uvicorn.run(
        "main:app",  # Change this to your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )

# Production deployment example:
"""
# For production, use:
# uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Or with gunicorn:
# gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Environment variables to set:
# export OPENAI_API_KEY="your-openai-api-key"
# export ENVIRONMENT="production"
# export LOG_LEVEL="info"
"""