"""
大模型智能体 - 使用DeepSeek API进行智能出牌决策
支持JSON格式输出，提高决策的结构化和可靠性
支持历史对话记忆，让LLM了解出牌过程
"""

import requests
import json
from typing import List, Dict, Any

class LLMAgent:
    """基于大模型的智能体"""
    
    def __init__(self, position: str, api_url: str = "https://api.deepseek.com/", 
                 model: str = "deepseek-chat", api_key: str = None):
        """
        初始化大模型智能体
        
        Args:
            position: 玩家位置 ('landlord', 'landlord_up', 'landlord_down')
            api_url: API接口地址
            model: 模型名称
            api_key: API密钥
        """
        self.name = 'LLM'
        self.position = position
        self.api_url = api_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}' if api_key else ''
        }
        
        # 历史对话记忆
        self.conversation_history = []
        
        # 卡牌转换映射
        self.EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                                8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q',
                                13: 'K', 14: 'A', 17: '2', 20: '小王', 30: '大王'}
        self.RealCard2EnvCard = {v: k for k, v in self.EnvCard2RealCard.items()}
    
    def reset_conversation_history(self):
        """重置历史对话（新局开始时调用）"""
        self.conversation_history = []
        print(f"LLM Agent [{self.get_position_name(self.position)}] 历史对话已重置")
    
    def add_to_history(self, role: str, content: str):
        """添加对话到历史记录"""
        self.conversation_history.append({"role": role, "content": content})
        # 限制历史记录长度，避免token过多
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def format_cards(self, cards: List[int]) -> str:
        """将环境卡牌格式化为可读字符串"""
        if not cards:
            return "过牌"
        return " ".join([self.EnvCard2RealCard.get(card, str(card)) for card in cards])
    
    def format_hand_cards(self, cards: List[int]) -> str:
        """格式化手牌为可读字符串"""
        from collections import Counter
        card_count = Counter(cards)
        result = []
        
        for card in sorted(card_count.keys()):
            card_name = self.EnvCard2RealCard.get(card, str(card))
            count = card_count[card]
            if count == 1:
                result.append(card_name)
            else:
                result.append(f"{card_name}×{count}")
        return " ".join(result)
    
    def get_position_name(self, position: str) -> str:
        """获取位置的中文名称"""
        position_names = {
            'landlord': '地主',
            'landlord_up': '地主上家',
            'landlord_down': '地主下家'
        }
        return position_names.get(position, position)
    
    def create_json_prompt_with_history(self, infoset) -> str:
        """创建包含历史对话的JSON格式提示词（优化版，减少token消耗）"""
        
        # 当前手牌
        hand_cards = infoset.player_hand_cards.copy()
        hand_cards_str = self.format_hand_cards(hand_cards)
        
        # 上家出牌
        last_move = infoset.last_move
        last_move_str = self.format_cards(last_move) if last_move else "无"
        
        # 可选动作（简化格式）
        legal_actions = infoset.legal_actions
        action_choices = []
        for i, action in enumerate(legal_actions):
            action_str = self.format_cards(action)
            action_choices.append(f"{i}: {action_str}")
        
        # 游戏状态信息
        position_name = self.get_position_name(self.position)
        
        # 简化历史记录 - 只保留最近5轮的关键决策
        history_summary = ""
        if self.conversation_history:
            recent_decisions = []
            for msg in reversed(self.conversation_history[-10:]):  # 最近10条消息
                if msg["role"] == "assistant" and "action_index" in msg.get("content", ""):
                    try:
                        decision = json.loads(msg["content"])
                        if "reason" in decision:
                            recent_decisions.append(decision["reason"][:30] + "...")
                    except:
                        continue
                
                if len(recent_decisions) >= 5:  # 只保留最近5个决策
                    break
            
            if recent_decisions:
                history_summary = "最近决策: " + " | ".join(reversed(recent_decisions)) + "\n"
        
        # 简化提示词，去除重复的规则说明
        prompt = f"""斗地主决策。位置: {position_name}。手牌: {hand_cards_str}。上家: {last_move_str}。

{history_summary}可选动作:
{"; ".join(action_choices)}

输出JSON格式: {{"action_index": 0, "reason": "简要理由", "confidence": 0.8}}

策略要点:
{"- 先手出小牌试探" if not last_move else "- 压制用最小牌"}
{"- 有炸弹可留后手" if any(len(action) == 4 and len(set(action)) == 1 for action in legal_actions) else ""}
{"- 王炸谨慎使用" if any(action == [20, 30] for action in legal_actions) else ""}

选择最优动作:"""
        
        return prompt.strip()
    
    def call_llm_api_json(self, prompt: str) -> Dict[str, Any]:
         """调用大模型API，使用JSON格式输出"""
         try:
             # 构建包含历史对话的消息列表
             messages = [
                 {
                     "role": "system", 
                     "content": "你是一位专业的斗地主玩家，只返回JSON格式的决策结果。保持理由简洁明了。"
                 }
             ]
             
             # 添加历史对话 - 限制历史记录长度
             recent_history = self.conversation_history[-6:] if len(self.conversation_history) > 6 else self.conversation_history
             messages.extend(recent_history)
             
             # 添加当前提示词
             messages.append({
                 "role": "user",
                 "content": prompt
             })
             
             payload = {
                 "model": self.model,
                 "messages": messages,
                 "response_format": {"type": "json_object"},
                 "temperature": 0.4,
                 "max_tokens": 150  # 减少max_tokens限制
             }
             
             response = requests.post(
                 f"{self.api_url}/chat/completions",
                 headers=self.headers,
                 json=payload,
                 timeout=30
             )
             
             if response.status_code == 200:
                 result = response.json()
                 content = result['choices'][0]['message']['content'].strip()
                 
                 # 添加到历史记录 - 保持精简
                 self.add_to_history("user", prompt)
                 self.add_to_history("assistant", content)
                 
                 # 解析JSON响应
                 try:
                     decision = json.loads(content)
                     return decision
                 except json.JSONDecodeError as e:
                     print(f"JSON解析失败: {e}")
                     print(f"响应内容: {content}")
                     return None
             else:
                 print(f"API调用失败: {response.status_code} - {response.text}")
                 return None
                 
         except Exception as e:
             print(f"调用大模型API时发生错误: {e}")
             return None
    
    def parse_json_response(self, decision: Dict[str, Any], legal_actions: List[List[int]]) -> int:
        """解析JSON格式的大模型响应"""
        if not decision or 'action_index' not in decision:
            return 0  # 默认选择第一个动作（通常是过牌）
        
        try:
            action_idx = int(decision['action_index'])
            if 0 <= action_idx < len(legal_actions):
                # 如果有reason字段，打印决策理由
                if 'reason' in decision:
                    reason = decision['reason']
                    # 截断过长的理由，避免输出混乱
                    if len(reason) > 100:
                        reason = reason[:100] + "..."
                    print(f"LLM决策理由: {reason}")
                return action_idx
            else:
                print(f"动作索引{action_idx}超出范围，使用默认值0")
                return 0
                
        except (ValueError, TypeError) as e:
            print(f"解析JSON决策时发生错误: {e}")
            return 0
    
    def act(self, infoset) -> List[int]:
        """大模型决策动作，使用JSON格式和历史对话"""
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]
        
        # 创建包含历史对话的JSON格式提示词
        prompt = self.create_json_prompt_with_history(infoset)
        
        # 调用大模型API获取JSON格式决策
        decision = self.call_llm_api_json(prompt)
        
        if decision is None:
            # API调用失败，使用备用策略（选择第一个合法动作）
            return infoset.legal_actions[0]
        
        # 解析JSON格式的决策
        action_idx = self.parse_json_response(decision, infoset.legal_actions)
        
        return infoset.legal_actions[action_idx]