from typing import Dict, List, TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Define input models for tools
class UserInfoInput(BaseModel):
    exercise_history: int
    age: int
    weight: float
    height: float
    injuries: List[str]
    goals: List[str]
    available_equipment: List[str]
    preferred_workout_time: str = "morning"
    weekly_workout_days: int = 3  # 주당 운동 일수

class AnalysisInput(BaseModel):
    fitness_level: str
    focus_areas: List[str]
    restrictions: List[str]
    preferred_workout_time: str
    weekly_workout_days: int

class ExerciseRecommendationInput(BaseModel):
    fitness_level: str
    goals: List[str]
    age: int
    weight: float
    height: float
    injuries: List[str]
    available_equipment: List[str]
    target_muscles: str = None  # Optional, for strength training

class ExerciseResearchData(BaseModel):
    exercise_name: str
    research_papers: List[Dict]
    clinical_trials: List[Dict]
    meta_analysis: List[Dict]

# Define types for our state
class WorkoutState(TypedDict):
    messages: Sequence[HumanMessage | AIMessage | ToolMessage]
    user_info: Dict
    workout_plan: Dict
    current_step: str
    feedback: Dict
    progress: Dict

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

# 운동 과학 연구 데이터베이스 (예시)
EXERCISE_RESEARCH_DB = {
    "스쿼트": {
        "research_papers": [
            {
                "title": "The Effect of Squat Depth on Lower Limb Muscle Activation",
                "authors": "Smith et al.",
                "year": 2022,
                "key_findings": "깊은 스쿼트가 대퇴사두근 활성화를 30% 더 증가",
                "recommendations": "무릎이 90도 이상 굽힐 때까지 스쿼트 수행"
            }
        ],
        "clinical_trials": [
            {
                "title": "Squat vs. Leg Press for Knee Rehabilitation",
                "participants": 100,
                "duration": "12주",
                "results": "스쿼트가 무릎 안정성 향상에 더 효과적"
            }
        ]
    },
    "데드리프트": {
        "research_papers": [
            {
                "title": "Deadlift Variations and Lower Back Health",
                "authors": "Johnson et al.",
                "year": 2023,
                "key_findings": "루마니안 데드리프트가 요통 환자에게 더 안전",
                "recommendations": "초기에는 루마니안 변형 추천"
            }
        ]
    }
}

# Define tools for the agent
@tool(args_schema=UserInfoInput)
def analyze_user_info(exercise_history: int, age: int, weight: float, height: float, 
                     injuries: List[str], goals: List[str], available_equipment: List[str],
                     preferred_workout_time: str = "morning", weekly_workout_days: int = 3) -> Dict:
    """사용자 정보를 분석하여 적절한 운동 강도와 중점 영역을 결정"""
    bmi = weight / ((height/100) ** 2)
    
    # 목표에 따른 운동 유형 결정
    if "다이어트" in goals or "체중 감량" in goals:
        focus_areas = ["cardio", "full_body"]
        workout_type = "weight_loss"
    elif "근력" in goals and exercise_history > 1:
        focus_areas = ["strength", "muscle_gain"]
        workout_type = "strength"
    else:
        focus_areas = ["full_body", "cardio"]
        workout_type = "general"
    
    return {
        "fitness_level": "intermediate" if exercise_history > 1 else "beginner",
        "focus_areas": focus_areas,
        "restrictions": injuries,
        "bmi": bmi,
        "preferred_workout_time": preferred_workout_time,
        "weekly_workout_days": weekly_workout_days,
        "workout_type": workout_type
    }

@tool(args_schema=ExerciseRecommendationInput)
def recommend_exercises(fitness_level: str, goals: List[str], age: int, weight: float, 
                       height: float, injuries: List[str], available_equipment: List[str],
                       target_muscles: str = None) -> Dict:
    """학술 연구를 바탕으로 사용자의 목표와 신체 조건에 맞는 운동을 추천"""
    # 연구 데이터 기반 추천 로직
    research_based_recommendations = []
    for exercise, research_data in EXERCISE_RESEARCH_DB.items():
        # 부상 고려
        if any(injury in str(research_data) for injury in injuries):
            continue
            
        # 연구 결과 분석
        research_summary = {
            "exercise": exercise,
            "evidence": [],
            "safety_considerations": []
        }
        
        # 연구 논문 분석
        for paper in research_data.get("research_papers", []):
            research_summary["evidence"].append({
                "type": "research_paper",
                "title": paper["title"],
                "key_findings": paper["key_findings"],
                "recommendations": paper["recommendations"]
            })
        
        # 임상 시험 분석
        for trial in research_data.get("clinical_trials", []):
            research_summary["evidence"].append({
                "type": "clinical_trial",
                "title": trial["title"],
                "results": trial["results"],
                "participants": trial["participants"]
            })
        
        research_based_recommendations.append(research_summary)
    
    # LLM을 사용한 추가 추천
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 운동 과학 전문가입니다. 다음 기준에 따라 운동을 추천해주세요:
1. 사용자의 목표와 신체 조건을 고려
2. 제공된 연구 결과를 참고하여 추천
3. 부상 위험을 최소화
4. 사용 가능한 장비를 고려
5. 각 운동의 효과와 적절한 강도를 구체적으로 설명

추천 형식:
{{
    "exercises": [
        {{
            "name": "운동 이름",
            "sets": "세트 수",
            "reps": "반복 횟수",
            "rest": "휴식 시간",
            "intensity": "강도 설명",
            "benefits": ["효과1", "효과2"],
            "cautions": ["주의사항1", "주의사항2"],
            "progression": "진행 방법",
            "research_evidence": ["연구 근거1", "연구 근거2"]
        }}
    ],
    "scientific_basis": "연구 근거 설명",
    "recommendations": "추가 권장사항"
}}"""),
        ("human", """다음 정보와 연구 결과를 바탕으로 운동을 추천해주세요:
- 운동 경력: {fitness_level}
- 목표: {goals}
- 나이: {age}
- 체중: {weight}kg
- 키: {height}cm
- 부상/제한사항: {injuries}
- 사용 가능한 장비: {available_equipment}
- 타겟 근육: {target_muscles}

연구 결과:
{research_data}""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "fitness_level": fitness_level,
        "goals": goals,
        "age": age,
        "weight": weight,
        "height": height,
        "injuries": injuries,
        "available_equipment": available_equipment,
        "target_muscles": target_muscles,
        "research_data": json.dumps(research_based_recommendations, ensure_ascii=False)
    })
    
    try:
        recommendations = json.loads(response.content)
        # 연구 데이터와 LLM 추천을 결합
        for exercise in recommendations["exercises"]:
            research_data = next(
                (r for r in research_based_recommendations if r["exercise"] == exercise["name"]),
                None
            )
            if research_data:
                exercise["research_evidence"] = [
                    f"{e['key_findings']} ({e['title']})"
                    for e in research_data["evidence"]
                    if e["type"] == "research_paper"
                ]
        return recommendations
    except:
        return {
            "exercises": [],
            "scientific_basis": "추천 실패",
            "recommendations": "다시 시도해주세요"
        }

@tool(args_schema=AnalysisInput)
def generate_workout_plan(fitness_level: str, focus_areas: List[str], restrictions: List[str],
                         preferred_workout_time: str, weekly_workout_days: int) -> Dict:
    """분석 결과를 바탕으로 개인화된 운동 계획 생성"""
    # 운동 강도 설정
    intensity_levels = {
        "beginner": {
            "sets": 3,
            "reps": "10-12",
            "rest": "60-90초",
            "cardio_duration": "20-30분"
        },
        "intermediate": {
            "sets": 4,
            "reps": "8-12",
            "rest": "45-60초",
            "cardio_duration": "30-40분"
        },
        "advanced": {
            "sets": 5,
            "reps": "6-12",
            "rest": "30-45초",
            "cardio_duration": "40-50분"
        }
    }
    
    # 주간 운동 계획 생성
    weekly_schedule = {}
    workout_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    
    # 운동 유형에 따른 계획 생성
    if focus_areas[0] == "cardio":
        # 다이어트/체중 감량 중심
        for i in range(weekly_workout_days):
            day = workout_days[i]
            cardio_recommendations = recommend_exercises.invoke({
                "fitness_level": fitness_level,
                "goals": ["체중 감량", "지구력 향상"],
                "age": 30,  # 실제 사용자 정보로 대체 필요
                "weight": 70,  # 실제 사용자 정보로 대체 필요
                "height": 170,  # 실제 사용자 정보로 대체 필요
                "injuries": restrictions,
                "available_equipment": ["맨몸운동", "자전거", "러닝"],
                "target_muscles": None
            })
            
            weekly_schedule[day] = {
                "type": "cardio",
                "exercises": cardio_recommendations["exercises"],
                "scientific_basis": cardio_recommendations["scientific_basis"],
                "recommendations": cardio_recommendations["recommendations"]
            }
    elif focus_areas[0] == "strength":
        # 근력 중심
        muscle_groups = ["chest", "back", "shoulders", "arms", "legs", "core"]
        for i in range(weekly_workout_days):
            day = workout_days[i]
            target_muscles = muscle_groups[i % len(muscle_groups)]
            strength_recommendations = recommend_exercises.invoke({
                "fitness_level": fitness_level,
                "goals": ["근력 향상", "근육량 증가"],
                "age": 30,  # 실제 사용자 정보로 대체 필요
                "weight": 70,  # 실제 사용자 정보로 대체 필요
                "height": 170,  # 실제 사용자 정보로 대체 필요
                "injuries": restrictions,
                "available_equipment": ["덤벨", "밴드", "맨몸운동"],
                "target_muscles": target_muscles
            })
            
            weekly_schedule[day] = {
                "type": "strength",
                "target_muscles": target_muscles,
                "exercises": strength_recommendations["exercises"],
                "scientific_basis": strength_recommendations["scientific_basis"],
                "recommendations": strength_recommendations["recommendations"]
            }
    else:
        # 전신 운동 중심
        for i in range(weekly_workout_days):
            day = workout_days[i]
            full_body_recommendations = recommend_exercises.invoke({
                "fitness_level": fitness_level,
                "goals": ["전신 운동", "기초 체력 향상"],
                "age": 30,  # 실제 사용자 정보로 대체 필요
                "weight": 70,  # 실제 사용자 정보로 대체 필요
                "height": 170,  # 실제 사용자 정보로 대체 필요
                "injuries": restrictions,
                "available_equipment": ["맨몸운동", "밴드"],
                "target_muscles": None
            })
            
            weekly_schedule[day] = {
                "type": "full_body",
                "exercises": full_body_recommendations["exercises"],
                "scientific_basis": full_body_recommendations["scientific_basis"],
                "recommendations": full_body_recommendations["recommendations"]
            }
    
    return {
        "weekly_schedule": weekly_schedule,
        "intensity": intensity_levels[fitness_level],
        "focus_areas": focus_areas,
        "preferred_workout_time": preferred_workout_time,
        "weekly_workout_days": weekly_workout_days
    }

@tool
def adjust_plan_based_on_feedback(workout_plan: Dict, feedback: Dict) -> Dict:
    """사용자 피드백을 바탕으로 운동 계획 조정"""
    adjusted_plan = workout_plan.copy()
    
    if feedback.get("too_difficult"):
        # 운동 강도 감소
        adjusted_plan["intensity"]["sets"] -= 1
        adjusted_plan["intensity"]["reps"] = "12-15"
        adjusted_plan["intensity"]["rest"] = "90-120초"
        if "cardio_duration" in adjusted_plan["intensity"]:
            adjusted_plan["intensity"]["cardio_duration"] = "15-20분"
    
    if feedback.get("too_easy"):
        # 운동 강도 증가
        adjusted_plan["intensity"]["sets"] += 1
        adjusted_plan["intensity"]["reps"] = "6-10"
        adjusted_plan["intensity"]["rest"] = "30-45초"
        if "cardio_duration" in adjusted_plan["intensity"]:
            adjusted_plan["intensity"]["cardio_duration"] = "40-50분"
    
    return adjusted_plan

def analyze_user(state: WorkoutState) -> WorkoutState:
    """사용자 정보 분석"""
    user_info = state["user_info"]
    analysis = analyze_user_info.invoke({
        "exercise_history": user_info["exercise_history"],
        "age": user_info["age"],
        "weight": user_info["weight"],
        "height": user_info["height"],
        "injuries": user_info["injuries"],
        "goals": user_info["goals"],
        "available_equipment": user_info["available_equipment"],
        "preferred_workout_time": user_info.get("preferred_workout_time", "morning"),
        "weekly_workout_days": user_info.get("weekly_workout_days", 3)
    })
    
    # AI 메시지 추가
    state["messages"].append(
        AIMessage(
            content=f"사용자 분석 완료: {analysis['fitness_level']} 레벨, {', '.join(analysis['focus_areas'])} 중점",
            additional_kwargs={
                "tool_calls": [{
                    "id": "analysis_1",
                    "type": "function",
                    "function": {
                        "name": "analyze_user_info",
                        "arguments": json.dumps(analysis, ensure_ascii=False)
                    }
                }]
            }
        )
    )
    
    # Tool 응답 메시지 추가
    state["messages"].append(
        ToolMessage(
            content="분석 완료",
            name="analyze_user_info",
            tool_call_id="analysis_1"
        )
    )
    
    state["current_step"] = "analysis_complete"
    return state

def generate_plan(state: WorkoutState) -> WorkoutState:
    """초기 운동 계획 생성"""
    user_info = state["user_info"]
    analysis = analyze_user_info.invoke({
        "exercise_history": user_info["exercise_history"],
        "age": user_info["age"],
        "weight": user_info["weight"],
        "height": user_info["height"],
        "injuries": user_info["injuries"],
        "goals": user_info["goals"],
        "available_equipment": user_info["available_equipment"],
        "preferred_workout_time": user_info.get("preferred_workout_time", "morning"),
        "weekly_workout_days": user_info.get("weekly_workout_days", 3)
    })
    
    workout_plan = generate_workout_plan.invoke({
        "fitness_level": analysis["fitness_level"],
        "focus_areas": analysis["focus_areas"],
        "restrictions": analysis["restrictions"],
        "preferred_workout_time": analysis["preferred_workout_time"],
        "weekly_workout_days": analysis["weekly_workout_days"]
    })
    
    # AI 메시지 추가
    state["messages"].append(
        AIMessage(
            content=f"운동 계획 생성 완료: {workout_plan['weekly_workout_days']}일/주, {workout_plan['focus_areas'][0]} 중심",
            additional_kwargs={
                "tool_calls": [{
                    "id": "plan_1",
                    "type": "function",
                    "function": {
                        "name": "generate_workout_plan",
                        "arguments": json.dumps(workout_plan, ensure_ascii=False)
                    }
                }]
            }
        )
    )
    
    # Tool 응답 메시지 추가
    state["messages"].append(
        ToolMessage(
            content="계획 생성 완료",
            name="generate_workout_plan",
            tool_call_id="plan_1"
        )
    )
    
    state["workout_plan"] = workout_plan
    state["current_step"] = "plan_generated"
    return state

def collect_feedback(state: WorkoutState) -> WorkoutState:
    """사용자 피드백 수집 및 계획 조정"""
    feedback = {
        "too_difficult": False,
        "too_easy": False,
        "preferred_exercises": [],
        "disliked_exercises": []
    }
    
    if feedback:
        adjusted_plan = adjust_plan_based_on_feedback.invoke({
            "workout_plan": state["workout_plan"],
            "feedback": feedback
        })
        
        # AI 메시지 추가
        state["messages"].append(
            AIMessage(
                content="운동 계획 조정 완료",
                additional_kwargs={
                    "tool_calls": [{
                        "id": "feedback_1",
                        "type": "function",
                        "function": {
                            "name": "adjust_plan_based_on_feedback",
                            "arguments": json.dumps(adjusted_plan, ensure_ascii=False)
                        }
                    }]
                }
            )
        )
        
        # Tool 응답 메시지 추가
        state["messages"].append(
            ToolMessage(
                content="계획 조정 완료",
                name="adjust_plan_based_on_feedback",
                tool_call_id="feedback_1"
            )
        )
        
        state["workout_plan"] = adjusted_plan
    
    state["feedback"] = feedback
    state["current_step"] = "feedback_received"
    return state

def finalize_plan(state: WorkoutState) -> WorkoutState:
    """최종 계획 검토 및 추천사항 제공"""
    # 간소화된 프롬프트 사용
    prompt = ChatPromptTemplate.from_messages([
        ("system", "운동 계획을 검토하고 핵심 추천사항을 2-3문장으로 제공해주세요."),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "운동 계획 검토: {workout_plan}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "messages": state["messages"],
        "workout_plan": json.dumps(state["workout_plan"], indent=2, ensure_ascii=False)
    })
    
    state["messages"].append(response)
    state["current_step"] = "complete"
    return state

# Create the graph
workflow = StateGraph(WorkoutState)

# Add nodes
workflow.add_node("analyze", analyze_user)
workflow.add_node("generate", generate_plan)
workflow.add_node("collect_feedback", collect_feedback)
workflow.add_node("finalize", finalize_plan)

# Add sequential edges
workflow.add_edge(START, "analyze")
workflow.add_edge("analyze", "generate")
workflow.add_edge("generate", "collect_feedback")
workflow.add_edge("collect_feedback", "finalize")
workflow.add_edge("finalize", END)

# Compile the graph
app = workflow.compile()

def generate_workout_routine(user_info: Dict) -> Dict:
    """
    Generate a personalized workout routine based on user information.
    """
    initial_state = {
        "messages": [],
        "user_info": user_info,
        "workout_plan": {},
        "feedback": {},
        "progress": {},
        "current_step": "start"
    }
    
    final_state = app.invoke(initial_state)
    
    # 메시지 내용 간소화
    serializable_state = {
        "messages": [
            {
                "type": msg.__class__.__name__,
                "content": msg.content,
                "additional_kwargs": {
                    k: v for k, v in msg.additional_kwargs.items() 
                    if v and not (isinstance(v, dict) and not v)
                } if hasattr(msg, "additional_kwargs") else {}
            }
            for msg in final_state["messages"]
        ],
        "workout_plan": final_state["workout_plan"],
        "current_step": final_state["current_step"]
    }
    
    return serializable_state

def is_workout_request(message: str) -> bool:
    """사용자 메시지가 운동 루틴 요청인지 확인"""
    workout_keywords = [
        "운동", "루틴", "추천", "계획", "프로그램", "트레이닝",
        "workout", "routine", "recommend", "plan", "program", "training"
    ]
    return any(keyword in message.lower() for keyword in workout_keywords)

def chat_with_workout_agent():
    """운동 루틴 추천 챗봇"""
    print("안녕하세요! 운동 루틴 추천 챗봇입니다.")
    print("운동 루틴을 추천해드릴까요?")
    print("종료하시려면 'quit'를 입력하세요.")
    
    while True:
        user_input = input("\n사용자: ").strip()
        
        if user_input.lower() == 'quit':
            print("\n챗봇: 좋은 하루 되세요! 운동 루틴 추천 챗봇을 종료합니다.")
            break
            
        if is_workout_request(user_input):
            # 더미 사용자 정보 사용
            sample_user_info = {
                "exercise_history": 2,
                "age": 28,
                "weight": 75,
                "height": 180,
                "injuries": ["요통"],
                "goals": ["근력", "지구력"],
                "available_equipment": ["덤벨", "밴드", "맨몸운동"],
                "preferred_workout_time": "morning",
                "weekly_workout_days": 4
            }
            
            try:
                result = generate_workout_routine(sample_user_info)
                
                # 결과 출력
                print("\n챗봇: 맞춤형 운동 루틴을 추천해드리겠습니다!")
                print("\n=== 주간 운동 계획 ===")
                
                # 주간 스케줄 출력
                for day, schedule in result["workout_plan"]["weekly_schedule"].items():
                    print(f"\n{day.upper()}:")
                    print(f"운동 유형: {schedule['type']}")
                    if "exercises" in schedule:
                        print("추천 운동:")
                        for exercise in schedule["exercises"]:
                            print(f"- {exercise['name']}")
                            print(f"  세트: {exercise['sets']}")
                            print(f"  반복: {exercise['reps']}")
                            print(f"  휴식: {exercise['rest']}")
                            if "research_evidence" in exercise:
                                print("  연구 근거:")
                                for evidence in exercise["research_evidence"]:
                                    print(f"  * {evidence}")
                
                print("\n추가 권장사항:")
                for exercise in result["workout_plan"]["weekly_schedule"]["monday"]["exercises"]:
                    if "recommendations" in exercise:
                        print(f"- {exercise['recommendations']}")
                
            except Exception as e:
                print(f"\n챗봇: 죄송합니다. 운동 루틴을 생성하는 중 오류가 발생했습니다: {str(e)}")
        else:
            print("\n챗봇: 죄송합니다. 운동 루틴 추천에 대해 말씀해 주시겠어요?")
            print("예시: '운동 루틴 추천해줘', '트레이닝 프로그램 만들어줘'")

if __name__ == "__main__":
    chat_with_workout_agent()

from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    pass