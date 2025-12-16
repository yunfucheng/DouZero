"""
大模型智能体 - 使用DeepSeek API进行智能出牌决策
支持JSON格式输出，提高决策的结构化和可靠性
支持历史对话记忆，让LLM了解出牌过程
"""

import requests
import json
import os
from collections import Counter
from typing import List, Dict, Any

class LLMAgent:
    """基于大模型的智能体"""
    
    def __init__(self, position: str, api_url: str = "https://api.deepseek.com/v3.2_speciale_expires_on_20251215",  # https://api.deepseek.com/
                 # model: str = "deepseek-chat", api_key: str = None):
                 model: str = "deepseek-reasoner", api_key: str = None):
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
        # 优先使用传入的key，否则尝试从环境变量获取
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}' if self.api_key else ''
        }
        
        # 调试模式
        self.debug_mode = True
        
        # 游戏ID和回合计数器
        self.game_id = None
        self.round_count = 0
        self.log_file_path = None
        
        # 游戏状态记录（替代历史对话）
        self.init_game_state(position)
        
        # 卡牌转换映射
        self.EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                                8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q',
                                13: 'K', 14: 'A', 17: '2', 20: '小王', 30: '大王'}
        self.RealCard2EnvCard = {v: k for k, v in self.EnvCard2RealCard.items()}
    

    def init_game_state(self, position: str):
        """初始化游戏状态"""
        self.game_state = {
            'position': position,
            'position_name': self.get_position_name(position),
            'hand_cards': [],  # 当前手牌
            'last_move': None,  # 上家出牌
            'last_player': None,  # 上一个出牌的玩家
            'played_cards': [],  # 已出牌记录（只存牌）
            'played_cards_with_player': [], # 详细出牌记录 (player, cards)
            'player_card_counts': {},  # 各玩家剩余牌数
            'recent_decisions': [],  # 最近决策记录
            'game_history': []  # 游戏历史记录
        }

    def format_cards(self, cards: List[int]) -> str:
        """将环境卡牌格式化为可读字符串，如果是已出牌列表则进行压缩"""
        if not cards:
            return "过牌"
        
        # 处理嵌套列表的情况（如played_cards）
        if cards and isinstance(cards[0], list):
            # 展平列表
            flat_cards = []
            for sublist in cards:
                if sublist:
                    flat_cards.extend(sublist)
            cards = flat_cards
            
            # 如果是已出牌（通常很长），使用压缩格式
            return self.format_hand_cards(cards)
        
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
    
    def format_hand_cards_compact(self, cards: List[int]) -> str:
        """格式化手牌为紧凑格式，不使用×符号和空格"""
        from collections import Counter
        
        if not cards:
            return ""
        
        card_counter = Counter(cards)
        result = []
        
        # 按牌值从小到大排序
        for card_value in sorted(card_counter.keys()):
            count = card_counter[card_value]
            card_name = self.EnvCard2RealCard.get(card_value, str(card_value))
            
            # 直接重复添加牌名，不使用×符号
            for _ in range(count):
                result.append(card_name)
        
        return "".join(result)  # 不加空格，直接连接
    
    def get_position_name(self, position: str) -> str:
        """获取位置的中文名称"""
        position_names = {
            'landlord': '地主',
            'landlord_up': '地主上家',
            'landlord_down': '地主下家'
        }
        return position_names.get(position, position)
    
    def start_new_game(self, game_id: str = None):
        """开始新游戏，初始化日志文件"""
        from datetime import datetime
        
        # 生成游戏ID（如果未提供）
        if game_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            position_name = self.get_position_name(self.position)
            game_id = f"{position_name}_{timestamp}"
        
        self.game_id = game_id
        self.round_count = 0
        
        # 重置游戏状态
        self.init_game_state(self.position)
        
        # 创建debug目录（如果不存在）
        debug_dir = "debug_logs"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        # 设置日志文件路径，使用 jsonl 格式
        self.log_file_path = f"{debug_dir}/game_log_{self.game_id}.jsonl"
        
        # 初始化日志文件，写入头部信息
        try:
            with open(self.log_file_path, "w", encoding="utf-8") as f:
                header = {
                    "type": "meta",
                    "game_id": self.game_id,
                    "position": self.position,
                    "position_name": self.get_position_name(self.position),
                    "start_time": datetime.now().isoformat()
                }
                f.write(json.dumps(header, ensure_ascii=False) + "\n")
            
            print(f"游戏日志已初始化: {self.log_file_path}")
        except Exception as e:
            print(f"初始化游戏日志失败: {e}")
            self.log_file_path = None
    
    def _debug_messages_to_file(self, messages: List[Dict[str, Any]]):
        """将messages输出到游戏日志文件"""
        if not self.debug_mode or self.log_file_path is None:
            return
            
        try:
            from datetime import datetime
            
            # 增加回合计数
            self.round_count += 1
            
            # 添加新的回合记录
            round_data = {
                "round": self.round_count,
                "timestamp": datetime.now().isoformat(),
                "type": "request",
                "messages": messages
            }
            
            # 追加到文件
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(round_data, ensure_ascii=False) + "\n")
            
        except Exception as e:
            print(f"保存调试信息时出错: {e}")
            
    def _debug_llm_response_to_file(self, response: str):
        """将LLM响应追加到游戏日志文件"""
        if not self.debug_mode or self.log_file_path is None:
            return
            
        try:
            from datetime import datetime
            
            # 添加响应记录
            round_data = {
                "round": self.round_count,
                "timestamp": datetime.now().isoformat(),
                "type": "response",
                "response": response
            }
            
            # 追加到文件
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(round_data, ensure_ascii=False) + "\n")
                 
            # print(f"LLM响应已保存到游戏日志: 回合{self.round_count}")
        except Exception as e:
            print(f"写入LLM响应到调试文件失败: {e}")

    def _debug_forced_move_to_file(self, action: List[int]):
        """记录强制动作（只有一个合法动作）到日志"""
        if not self.debug_mode or self.log_file_path is None:
            return
            
        try:
            from datetime import datetime
            
            # 增加回合计数
            self.round_count += 1
            
            # 格式化动作
            action_str = self.format_cards(action)
            
            # 添加记录
            round_data = {
                "round": self.round_count,
                "timestamp": datetime.now().isoformat(),
                "type": "forced_move",
                "action": action_str,
                "reason": "唯一合法动作"
            }
            
            # 追加到文件
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(round_data, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"写入强制动作日志失败: {e}")
    
    def create_comprehensive_prompt(self, infoset) -> str:
        """创建包含全局游戏状态的提示词"""
        # 更新游戏状态
        self._update_game_state(infoset)
        
        # 当前手牌（使用详细格式，例如 3×3）
        hand_cards_str = self.format_hand_cards(self.game_state['hand_cards'])
        
        # 上家出牌
        last_move_str = self.format_cards(self.game_state['last_move']) if self.game_state['last_move'] else "无"
        
        # 构建历史对局记录（全量历史，使用缩写）
        history = ""
        if self.game_state.get('played_cards_with_player'):
            # 完整历史记录
            all_plays = self.game_state['played_cards_with_player']
            history_parts = []
            
            # 角色缩写映射
            role_abbr = {
                'landlord': 'L',      # Landlord
                'landlord_up': 'U',   # Up
                'landlord_down': 'D'  # Down
            }
            
            for i, (player, play) in enumerate(all_plays):
                # 获取角色缩写，如果未知则标记为?
                abbr = role_abbr.get(player, '?')
                play_str = self.format_cards(play) if play else "过"
                round_num = i + 1
                history_parts.append(f"{round_num:02d}{abbr}: {play_str}")
            
            # 格式化历史记录，使其更紧凑但易读
            history = "\n历史对局([序号][地主L|上家U|下家D]): " + ", ".join(history_parts)
        
        # 构建合法动作列表
        legal_actions = []
        for action in infoset.legal_actions:
            if action:  # 非空动作
                legal_actions.append(self.format_cards(action))
        
        legal_actions_str = ""
        if legal_actions:
            legal_actions_str = f"合法动作选项: {', '.join(legal_actions)}"
        
        # 未知牌字段 - 显示剩余未知的牌（其他两家的手牌集合）
        unknown_cards_str = self._calculate_unknown_cards_str()
        
        # 构建完整提示词
        
        # 1. 角色与目标
        role_desc = ""
        if self.position == 'landlord':
            role_desc = "你是【地主】。目标：打完手中所有牌。你需要独自对抗两家农民。"
        elif self.position == 'landlord_up':
            role_desc = "你是【地主上家】（农民）。目标：你或你的队友（地主下家）先打完牌。策略：主要任务是顶住地主的牌，不让地主过小牌，必要时牺牲自己保队友。"
        elif self.position == 'landlord_down':
            role_desc = "你是【地主下家】（农民）。目标：你或你的队友（地主上家）先打完牌。策略：地主上家会限制地主，你需要伺机跑牌，或者在地主上家顶不住时接力。"
            
        # 2. 剩余牌数分析
        card_counts_info = ""
        if self.game_state.get('player_card_counts'):
            counts = []
            for pos, count in self.game_state['player_card_counts'].items():
                name = self.get_position_name(pos)
                counts.append(f"{name}: {count}张")
            card_counts_info = " | ".join(counts)
            
        # 3. 关键牌分析 (2, 王)
        key_cards_analysis = ""
        # 简单的剩余大牌分析已经在 _calculate_remaining_cards 中实现，这里可以复用或增强
        # 这里直接使用 _calculate_unknown_cards_str 返回的结果，并在prompt中引导LLM注意
        
        prompt = f"""角色: {role_desc}
当前位置: {self.game_state['position_name']}
各家剩余手牌数: {card_counts_info}

【你的手牌】: {hand_cards_str}

【当前牌局状态】
上家出牌: {last_move_str}
已出牌记录: {self.format_cards(self.game_state['played_cards']) if self.game_state['played_cards'] else "无"}
未知牌（对手/队友手中的牌）: {unknown_cards_str}

{history}

【合法动作选项】
{legal_actions_str}

【思考与决策】
请分析当前局势，考虑以下因素：
1. 你的角色职责（地主进攻/农民配合）。
2. 对手剩余牌数（特别是如果有人只剩1-2张牌时，必须全力阻截）。
3. 记牌：大王、小王、2、A是否已出完？
4. 如果你是农民，你的队友是谁？不要压队友的牌，除非为了过牌权的必要转换。

输出JSON格式: {{"cards": "3 3 3", "reason": "你的分析与决策理由", "confidence": 0.8}}
注意：cards字段必须完全匹配上述合法动作选项之一。如果是过牌，cards字段填"过牌"。

请根据以上信息，给出最佳出牌决策。"""
        
        return prompt.strip()
    
    def _calculate_unknown_cards_str(self) -> str:
        """计算未知牌（全集 - 手牌 - 已出牌）"""
        # 初始化一副完整的牌
        full_deck = []
        # 3-14(A) 各4张
        for i in range(3, 15):
            full_deck.extend([i] * 4)
        # 17(2) 4张
        full_deck.extend([17] * 4)
        # 大小王
        full_deck.extend([20, 30])
        
        deck_counter = Counter(full_deck)
        
        # 减去手牌
        hand_counter = Counter(self.game_state['hand_cards'])
        deck_counter -= hand_counter
        
        # 减去已出牌
        played_cards_flat = []
        for move in self.game_state['played_cards']:
            if move:
                played_cards_flat.extend(move)
        played_counter = Counter(played_cards_flat)
        deck_counter -= played_counter
        
        # 转换回列表并排序
        unknown_cards = []
        for card, count in deck_counter.items():
            if count > 0:
                unknown_cards.extend([card] * count)
        
        unknown_cards.sort()
        
        return self.format_hand_cards_compact(unknown_cards)

    def _update_game_state(self, infoset):
        """更新游戏状态"""
        # 更新手牌
        self.game_state['hand_cards'] = infoset.player_hand_cards.copy()
        
        # 更新上家出牌
        if hasattr(infoset, 'last_move') and infoset.last_move:
            self.game_state['last_move'] = infoset.last_move
            
        # 更新玩家牌数（如果可用）
        if hasattr(infoset, 'player_card_counts'):
            self.game_state['player_card_counts'] = infoset.player_card_counts.copy()
        
        # 更新上一个出牌的玩家 - 使用infoset的last_pid属性
        if hasattr(infoset, 'last_pid') and infoset.last_pid:
            self.game_state['last_player'] = infoset.last_pid
        elif hasattr(infoset, 'last_player'):
            self.game_state['last_player'] = infoset.last_player
            
        # 重建完整的历史记录（如果可用）
        # 利用 infoset.card_play_action_seq 推导历史
        if hasattr(infoset, 'card_play_action_seq'):
            self._reconstruct_history(infoset.card_play_action_seq)
        else:
            # 兼容旧逻辑：只记录上家出牌
             if hasattr(infoset, 'last_move') and infoset.last_move:
                 if infoset.last_move not in self.game_state['played_cards']:
                     self.game_state['played_cards'].append(infoset.last_move)

    def _reconstruct_history(self, action_seq):
        """根据动作序列重建历史记录（推导每一步的玩家）"""
        if not action_seq:
            return
            
        # 斗地主出牌顺序：地主 -> 地主下家 -> 地主上家
        players = ['landlord', 'landlord_down', 'landlord_up']
        
        # 记录已处理的动作数量，避免重复计算
        current_len = len(self.game_state.get('played_cards_with_player', []))
        if len(action_seq) <= current_len:
            # 如果序列长度没变或变短（新游戏？），可能需要重置或跳过
            if len(action_seq) < current_len:
                 # 序列变短，说明是新游戏，重置
                 self.game_state['played_cards_with_player'] = []
                 self.game_state['played_cards'] = []
                 current_len = 0
            else:
                 return
        
        # 增量更新
        new_actions = action_seq[current_len:]
        
        # 确定起始玩家
        # 我们可以根据当前已有的记录数量来推导下一个玩家
        # 第0手牌是地主出的
        
        for i, action in enumerate(new_actions):
            global_idx = current_len + i
            player_idx = global_idx % 3
            player = players[player_idx]
            
            # 记录详细历史 (player, cards)
            self.game_state['played_cards_with_player'].append((player, action))
            
            # 记录简单历史 (只存非空牌)
            if action:
                self.game_state['played_cards'].append(action)
    
    def _calculate_remaining_cards(self) -> str:
        """计算剩余牌情况"""
        # 统计已出牌
        played_cards = set()
        for move in self.game_state['played_cards']:
            if move:  # 确保move不为空
                played_cards.update(move)
        
        # 计算剩余大牌
        big_cards = []
        for card, name in self.EnvCard2RealCard.items():
            if card not in played_cards and card in [14, 17, 20, 30]:  # A, 2, 小王, 大王
                big_cards.append(name)
        
        if big_cards:
            return f"剩余大牌: {', '.join(big_cards)}"
        else:
            return "大牌已全部出完"
    
    def _get_played_big_cards(self) -> str:
        """获取已出的大牌"""
        played_big = []
        for move in self.game_state['played_cards']:
            if move:  # 确保move不为空
                for card in move:
                    if card in [14, 17, 20, 30]:  # A, 2, 小王, 大王
                        played_big.append(self.EnvCard2RealCard[card])
        
        if played_big:
            # 去重并排序
            unique_cards = list(set(played_big))
            return ", ".join(unique_cards)
        return ""
    
    def call_llm_api_json(self, prompt: str) -> Dict[str, Any]:
         """调用大模型API，使用JSON格式输出"""
         try:
             # 构建消息列表，不包含历史对话
             messages = [
                 {
                     "role": "system", 
                     "content": "你是一位专业的斗地主玩家，只返回JSON格式的决策结果。保持理由简洁明了。"
                 },
                 {
                     "role": "user",
                     "content": prompt
                 }
             ]
             
             # 将messages输出到文本文件，方便调试
             self._debug_messages_to_file(messages)
             
             payload = {
                "model": self.model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0.4,
                "max_tokens": 512  # 增加 max_tokens 以避免响应被截断
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
                 
                 # 打印LLM原始响应，便于调试
                 # print(f"LLM原始响应: {content}")
                 
                 # 将LLM响应也保存到日志文件
                 self._debug_llm_response_to_file(content)
                 
                 # 解析JSON响应
                 try:
                     decision = json.loads(content)
                     # 更新游戏状态中的最近决策
                     if 'reason' in decision:
                         reason = decision['reason']
                         # 截断过长的理由
                         if len(reason) > 50:
                             reason = reason[:50] + "..."
                         self.game_state['recent_decisions'].append(reason)
                         # 只保留最近5个决策
                         if len(self.game_state['recent_decisions']) > 5:
                             self.game_state['recent_decisions'] = self.game_state['recent_decisions'][-5:]
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
    
    def parse_card_response(self, decision: Dict[str, Any], legal_actions: List[List[int]]) -> int:
        """解析LLM输出的牌，匹配到对应的可选动作索引"""
        if not decision or 'cards' not in decision:
            return 0  # 默认选择第一个动作（通常是过牌）
        
        try:
            raw_cards = decision.get('cards', '')
            if isinstance(raw_cards, list):
                # 如果LLM返回了列表，将其转换为字符串
                cards_str = " ".join(str(c) for c in raw_cards)
            else:
                cards_str = str(raw_cards)
            
            cards_str = cards_str.strip()
            
            # 如果是过牌，返回过牌动作的索引
            if cards_str == "过牌" or cards_str.lower() == "pass":
                for i, action in enumerate(legal_actions):
                    if not action:  # 空列表表示过牌
                        return i
                return 0  # 如果没找到过牌动作，返回第一个
            
            # 将牌字符串转换为环境卡牌列表
            cards_list = []
            
            # 特殊处理大小王，因为split()可能会把"小王"分成"小"和"王"（如果用户用空格分隔不当）
            # 或者用户可能直接返回 "小王 大王"
            
            # 预处理：将中文逗号替换为空格，去除多余空白
            cards_str = cards_str.replace('，', ' ').strip()
            
            # 分词
            tokens = cards_str.split()
            
            for token in tokens:
                if token in self.RealCard2EnvCard:
                    cards_list.append(self.RealCard2EnvCard[token])
                else:
                    # 尝试处理可能的连写或特殊情况
                    # 比如 "小王大王" 连写的情况，或者 "333" 连写
                    # 这里暂时只处理标准空格分隔，如果遇到无法识别的token，可能需要更复杂的解析
                    # 鉴于Prompt要求空格分隔，我们先假设LLM遵循指令
                    # 但针对 "大王" 报错，可能是 split 逻辑没问题，而是其他原因
                    # 检查是否是 "大王" 本身就在 RealCard2EnvCard 中
                    # self.RealCard2EnvCard 包含 '大王': 30
                    pass
            
            # 再次检查：如果是单张大王/小王，tokens应该是 ['大王']
            # 如果 cards_str 是 "大王"，split后是 ['大王']，应该能匹配到
            
            # 调试信息：如果匹配失败，打印详细信息以便排查
            if not cards_list and cards_str:
                 # 可能是解析逻辑漏掉了某些情况
                 pass
            
            # 如果没有有效的牌，返回默认值
            if not cards_list:
                return 0
            
            # 在合法动作中查找匹配的动作
            for i, action in enumerate(legal_actions):
                # 转换为多重集比较，忽略顺序
                if Counter(action) == Counter(cards_list):
                    # 如果有reason字段，打印决策理由
                    if 'reason' in decision:
                        reason = decision['reason']
                        # 截断过长的理由，避免输出混乱
                        if len(reason) > 100:
                            reason = reason[:100] + "..."
                        # print(f"LLM决策理由: {reason}")
                    return i
            
            # 如果没找到匹配的动作，直接抛出异常
            raise RuntimeError(f"无法匹配动作: {cards_str}")
                
        except (ValueError, TypeError, AttributeError) as e:
            print(f"解析牌决策时发生错误: {e}")
            return 0
    
    def act(self, infoset) -> List[int]:
        """大模型决策动作，使用全局游戏状态"""
        # 无论如何，首先更新游戏状态，确保状态是最新的
        self._update_game_state(infoset)
        
        if len(infoset.legal_actions) == 1:
            # 如果只有一个合法动作，直接返回
            action = infoset.legal_actions[0]
            
            # 记录强制动作到日志，确保日志连贯性
            self._debug_forced_move_to_file(action)
            
            # _update_game_state已经处理了历史记录更新，这里不需要手动append
            return action
        
        # 创建包含全局游戏状态的提示词，让LLM直接输出要出的牌
        # 注意：create_comprehensive_prompt 内部也会调用 _update_game_state，但这没有副作用
        prompt = self.create_comprehensive_prompt(infoset)
        
        # 调用大模型API获取JSON格式决策
        decision = self.call_llm_api_json(prompt)
        
        if decision is None:
            # API调用失败，使用备用策略（选择第一个合法动作）
            action = infoset.legal_actions[0]
            # 更新已出牌记录
            if action and action not in self.game_state['played_cards']:
                self.game_state['played_cards'].append(action)
            return action
        
        # 解析LLM输出的牌，匹配到对应的可选动作索引
        try:
            action_idx = self.parse_card_response(decision, infoset.legal_actions)
        except RuntimeError as e:
            print(f"LLM非法动作警告: {e}。将使用默认动作(索引0)。")
            # 记录这次错误的决策以便后续分析（可选）
            # self._debug_llm_response_to_file(prompt, decision, infoset.legal_actions)
            action_idx = 0

        action = infoset.legal_actions[action_idx]
        
        # 更新已出牌记录
        if action and action not in self.game_state['played_cards']:
            self.game_state['played_cards'].append(action)
        
        return action