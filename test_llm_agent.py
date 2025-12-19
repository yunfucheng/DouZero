#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试LLM Agent的修复效果
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from douzero.evaluation.llm_agent import LLMAgent

class MockInfoset:
    """模拟Infoset对象"""
    def __init__(self):
        # 模拟地主的初始手牌
        self.player_hand_cards = [30, 20, 17, 17, 14, 13, 13, 12, 11, 10, 9, 9, 8, 7, 7, 7, 6, 5, 4]
        # 模拟上一次出牌
        self.last_move = [20]  # 小王
        # 模拟牌局动作序列
        self.card_play_action_seq = [
            [3, 3, 3],           # 地主出333
            [6, 6, 6],           # 下家出666
            [],                   # 上家过牌
            [11, 11, 11],         # 地主出JJJ
            [13, 13, 13],         # 下家出KKK
            [14, 14, 14],         # 上家出AAA
            [],                   # 地主过牌
            [],                   # 下家过牌
            [7],                  # 上家出7
            [],                   # 地主过牌
            [9],                  # 下家出9
            [],                   # 上家过牌
            [],                   # 地主过牌
            [4, 5, 5, 5, 5, 10],  # 下家出4555510
            [],                   # 上家过牌
            [],                   # 地主过牌
            [12],                 # 下家出Q
            [],                   # 上家过牌
            [],                   # 地主过牌
            [4],                  # 下家出4
            [11],                 # 上家出J
            [],                   # 地主过牌
            [17],                 # 下家出2
            [20]                  # 上家出小王
        ]
        # 模拟合法动作选项
        self.legal_actions = [
            [30],  # 大王
            []     # 过牌
        ]
        
        # 注意：我们故意不提供player_card_counts，测试手动计算功能


def test_llm_agent():
    """测试LLM Agent的修复效果"""
    print("=== 测试LLM Agent修复效果 ===\n")
    
    # 创建LLM Agent实例（地主位置）
    agent = LLMAgent(position='landlord')
    
    # 创建模拟的infoset
    infoset = MockInfoset()
    
    # 调用create_comprehensive_prompt方法生成提示词
    prompt = agent.create_comprehensive_prompt(infoset)
    
    # 打印生成的提示词，重点检查以下内容：
    # 1. 各家剩余手牌数是否正确
    # 2. 历史对局信息是否优化
    # 3. 牌局阶段判断是否正确
    print("生成的提示词：")
    print("=" * 50)
    print(prompt)
    print("=" * 50)
    
    # 检查关键信息
    print("\n=== 关键信息检查 ===")
    
    # 1. 检查剩余手牌数
    if "各家剩余手牌数:" in prompt:
        print("✅ 各家剩余手牌数已显示")
    else:
        print("❌ 各家剩余手牌数未显示")
    
    # 2. 检查历史对局统计信息
    if "【出牌统计】" in prompt:
        print("✅ 出牌统计信息已添加")
    else:
        print("❌ 出牌统计信息未添加")
    
    # 3. 检查历史对局按轮次分组
    if "【详细对局】" in prompt:
        print("✅ 详细对局信息已按轮次分组")
    else:
        print("❌ 详细对局信息未按轮次分组")
    
    # 4. 检查牌局阶段判断
    if "【牌局阶段：" in prompt:
        print("✅ 牌局阶段判断已添加")
    else:
        print("❌ 牌局阶段判断未添加")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_llm_agent()
