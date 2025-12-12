#!/usr/bin/env python3
"""
斗地主游戏可视化入口
集成现有的AI agent、规则agent，并添加human agent支持
与evaluate.py保持一致的命令行参数格式
"""

import argparse
import os
import pickle
import random
from collections import Counter

from douzero.env.env import Env
from douzero.evaluation.simulation import load_card_play_models

def format_cards(cards):
    """将环境卡牌格式化为可读字符串"""
    if not cards:
        return "过牌"
    
    EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                       8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q',
                       13: 'K', 14: 'A', 17: '2', 20: '小王', 30: '大王'}
    
    return " ".join([EnvCard2RealCard.get(card, str(card)) for card in cards])

def format_hand_cards(cards):
    """格式化手牌为可读字符串"""
    card_count = Counter(cards)
    result = []
    
    EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                       8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q',
                       13: 'K', 14: 'A', 17: '2', 20: '小王', 30: '大王'}
    
    for card in sorted(card_count.keys()):
        card_name = EnvCard2RealCard.get(card, str(card))
        count = card_count[card]
        if count == 1:
            result.append(card_name)
        else:
            result.append(f"{card_name}×{count}")
    return " ".join(result)

class HumanAgent:
    """人类玩家智能体"""
    
    def __init__(self, position):
        self.name = 'Human'
        self.position = position
    
    def act(self, infoset):
        """人类玩家输入动作"""
        legal_actions = infoset.legal_actions
        
        print(f"\n[{self.position}] 您的回合:")
        print(f"手牌: {format_hand_cards(infoset.player_hand_cards)}")
        
        if infoset.last_move:
            print(f"上家出牌: {format_cards(infoset.last_move)}")
        
        print("可选动作:")
        for i, action in enumerate(legal_actions):
            print(f"{i}: {format_cards(action)}")
        
        while True:
            try:
                choice = input(f"请选择动作 (0-{len(legal_actions)-1}): ").strip()
                action_idx = int(choice)
                if 0 <= action_idx < len(legal_actions):
                    return legal_actions[action_idx]
                else:
                    print(f"请输入 0-{len(legal_actions)-1} 之间的数字")
            except ValueError:
                print("请输入有效的数字")

def play_single_game(card_play_data, players, verbose=True):
    """玩一局游戏"""
    
    env = Env('wp')  # 使用wp目标
    env.reset()
    env._env.card_play_init(card_play_data)
    
    if verbose:
        print("=" * 60)
        print("新局开始！")
        
        # 显示手牌
        format_cards = lambda cards: " ".join([str(card) for card in cards])
        print(f"地主手牌: {format_cards(card_play_data['landlord'])}")
        print(f"地主上家手牌: {format_cards(card_play_data['landlord_up'])}")
        print(f"地主下家手牌: {format_cards(card_play_data['landlord_down'])}")
        print(f"地主底牌: {format_cards(env._env.three_landlord_cards)}")
        print("-" * 60)
    
    move_count = 0
    has_human = any(player.name == 'Human' for player in players.values())
    
    # 游戏主循环
    while not env._game_over:
        current_player = env._acting_player_position
        infoset = env._game_infoset
        
        # 获取玩家动作
        action = players[current_player].act(infoset)
        
        # 设置动作并执行
        env.players[current_player].set_action(action)
        env._env.step()
        move_count += 1
        
        if verbose:
            player_name = {
                'landlord': '地主',
                'landlord_up': '地主上家',
                'landlord_down': '地主下家'
            }[current_player]
            
            action_str = format_cards(action)
            remaining_cards = len(env._env.info_sets[current_player].player_hand_cards)
            agent_name = players[current_player].name
            
            print(f"第{move_count:2d}轮 - {player_name:6s}({agent_name:8s}): {action_str:15s} "
                  f"剩余{remaining_cards:2d}张")
    
    # 游戏结束，显示结果
    if verbose:
        print("-" * 60)
        winner = env._game_winner
        if winner == 'landlord':
            print(f"游戏结束！地主获胜！炸弹数: {env._game_bomb_num}")
        else:
            print(f"游戏结束！农民获胜！炸弹数: {env._game_bomb_num}")
        print("=" * 60)
        print()
    
    return {
        'winner': env._game_winner,
        'bomb_num': env._game_bomb_num,
        'move_count': move_count
    }

def generate_card_play_data():
    """生成一局随机的卡牌数据"""
    import numpy as np
    
    # 创建牌组
    deck = []
    for i in range(3, 15):
        deck.extend([i for _ in range(4)])
    deck.extend([17 for _ in range(4)])  # 2
    deck.extend([20, 30])  # 小王，大王
    
    # 洗牌
    np.random.shuffle(deck)
    
    # 分配卡牌
    card_play_data = {
        'landlord': deck[:20],
        'landlord_up': deck[20:37],
        'landlord_down': deck[37:54],
        'three_landlord_cards': deck[17:20],
    }
    
    # 排序手牌
    for key in card_play_data:
        card_play_data[key].sort()
    
    return card_play_data

def main():
    parser = argparse.ArgumentParser(
        description='Dou Dizhu Play Game - 斗地主游戏可视化入口')
    
    # 与evaluate.py保持一致的参数格式
    parser.add_argument('--landlord', type=str,
                       default='random',
                       help='地主智能体 (random, rlcard, human, llm, 或模型路径如baselines/douzero_ADP/landlord.ckpt)')
    parser.add_argument('--landlord_up', type=str,
                       default='random',
                       help='地主上家智能体 (random, rlcard, human, llm, 或模型路径)')
    parser.add_argument('--landlord_down', type=str,
                       default='random',
                       help='地主下家智能体 (random, rlcard, human, llm, 或模型路径)')
    parser.add_argument('--num_games', type=int, default=1,
                       help='游戏局数')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='显示详细输出')
    parser.add_argument('--stats_only', action='store_true', default=False,
                       help='只显示统计信息，不显示每局详情')
    parser.add_argument('--eval_data', type=str, default=None,
                       help='使用预设的评估数据文件，如果不指定则随机生成')
    parser.add_argument('--llm_api_key', type=str, default=None,
                       help='大模型API密钥（使用llm agent时必需）')
    parser.add_argument('--llm_api_url', type=str, default="https://api.deepseek.com/",
                       help='大模型API接口地址')
    parser.add_argument('--llm_model', type=str, default="deepseek-chat",
                       help='大模型名称')
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    # 创建玩家配置字典
    card_play_model_path_dict = {
        'landlord': args.landlord,
        'landlord_up': args.landlord_up,
        'landlord_down': args.landlord_down
    }
    
    # 检查是否有human玩家
    has_human = any(value == 'human' for value in card_play_model_path_dict.values())
    
    if has_human and args.num_games > 1:
        print("注意：有human玩家参与时，游戏局数自动设置为1")
        args.num_games = 1
    
    # 显示智能体配置
    print(f"智能体配置:")
    for pos, agent in card_play_model_path_dict.items():
        chinese_pos = {'landlord': '地主', 'landlord_up': '地主上家', 'landlord_down': '地主下家'}[pos]
        print(f"{chinese_pos}: {agent}")
    print()
    
    # 加载或生成游戏数据
    if args.eval_data:
        # 使用预设的评估数据
        with open(args.eval_data, 'rb') as f:
            card_play_data_list = pickle.load(f)
        print(f"使用评估数据文件: {args.eval_data} (共{len(card_play_data_list)}局)")
        
        # 限制游戏局数
        if args.num_games > len(card_play_data_list):
            print(f"游戏局数超过数据文件中的局数，自动调整为{len(card_play_data_list)}")
            args.num_games = len(card_play_data_list)
    else:
        # 随机生成游戏数据
        card_play_data_list = [generate_card_play_data() for _ in range(args.num_games)]
        print(f"随机生成{args.num_games}局游戏数据")
    
    # 创建LLM Agent实例池，实现实例复用
    llm_agents_pool = {}
    
    # 运行游戏
    results = []
    for game_id in range(args.num_games):
        if args.num_games > 1 and not args.stats_only:
            print(f"\n第 {game_id + 1} 局游戏:")
        
        # 为每局游戏创建新的玩家实例（避免状态污染），但LLM Agent复用实例
        players = {}
        
        # 分别加载每个位置的agent
        for pos in ['landlord', 'landlord_up', 'landlord_down']:
            agent_type = card_play_model_path_dict[pos]
            if agent_type == 'human':
                players[pos] = HumanAgent(pos)
            elif agent_type == 'random':
                from douzero.evaluation.random_agent import RandomAgent
                players[pos] = RandomAgent()
            elif agent_type == 'rlcard':
                from douzero.evaluation.rlcard_agent import RLCardAgent
                players[pos] = RLCardAgent(pos)
            elif agent_type == 'llm':
                from douzero.evaluation.llm_agent import LLMAgent
                if not args.llm_api_key:
                    print(f"错误：使用llm agent时必须提供--llm_api_key参数")
                    return
                
                # 复用LLM Agent实例，避免重复创建
                if pos not in llm_agents_pool:
                    llm_agents_pool[pos] = LLMAgent(pos, api_url=args.llm_api_url, 
                                                      model=args.llm_model, api_key=args.llm_api_key)
                
                players[pos] = llm_agents_pool[pos]
                
                # 新局开始时重置历史对话
                if game_id == 0:  # 第一局游戏
                    players[pos].reset_conversation_history()
                
            else:
                # 模型路径
                from douzero.evaluation.deep_agent import DeepAgent
                players[pos] = DeepAgent(pos, agent_type)
        
        result = play_single_game(
            card_play_data_list[game_id],
            players,
            verbose=not args.stats_only and (args.num_games == 1 or not args.stats_only)
        )
        results.append(result)
    
    # 显示统计信息
    if args.num_games > 1:
        landlord_wins = sum(1 for r in results if r['winner'] == 'landlord')
        farmer_wins = sum(1 for r in results if r['winner'] == 'farmer')
        avg_moves = sum(r['move_count'] for r in results) / len(results)
        avg_bombs = sum(r['bomb_num'] for r in results) / len(results)
        
        print(f"\n{'='*60}")
        print(f"统计信息 ({args.num_games} 局游戏):")
        print(f"地主胜率: {landlord_wins}/{args.num_games} ({landlord_wins/args.num_games*100:.1f}%)")
        print(f"农民胜率: {farmer_wins}/{args.num_games} ({farmer_wins/args.num_games*100:.1f}%)")
        print(f"平均每局回合数: {avg_moves:.1f}")

if __name__ == '__main__':
    main()