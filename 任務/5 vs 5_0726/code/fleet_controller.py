import numpy as np
import logging
from sample_agent import MyAgent
from pycmo.lib.actions import delete_unit, add_unit, set_unit_to_mission

MAX_STEP = 500         # 與 MyAgent.max_episode_steps 相同

class FleetController:
    def __init__(self, player_side: str, enemy_side: str, ship_names: list[str]):
        self.agents = [MyAgent(player_side, enemy_side, n) for n in ship_names]
        self.player_side = player_side
        self.enemy_side = enemy_side
        self.logger = logging.getLogger("Fleet")
        self.episode_id = 0
        self.reset_cmd = ""          # ★ 新增：確保屬性存在
        self._init_pos_cached = False          # 開局快照是否完成
        self._init_pos = {}                    # {(side, name): (dbid, lat, lon)}

    # 供 run_loop 初次載入
    def reset(self):
        # 只為了快照初始位置
        if hasattr(self, "_first_features_snapshot"):
            self._cache_init_positions(self._first_features_snapshot)
        return "\n".join([ag.reset() for ag in self.agents])

    def _cache_init_positions(self, features):
        if self._init_pos_cached:
            return
        for side in (self.player_side, self.enemy_side):
            for u in features.units[side]:
                self._init_pos[(side, u.Name)] = (u.DBID, u.Lat, u.Lon)
        self._init_pos_cached = True

    # 主迴圈呼叫
    def action(self, features):
        # ★ 第一句就把開局座標存起來
        self._cache_init_positions(features)
        
        cmds = []
        all_done  = True
        time_over = False

        for ag in self.agents:
            cmds.append(ag.action(features))
            all_done  &= ag.completed
            if ag.episode_step >= MAX_STEP:
                time_over = True

        # → 仍有艦未抵達 & 未超時，直接回傳指令
        if not all_done and not time_over:
            return "\n".join(cmds)

        # ---------- 統計區域 ----------
        for ag in self.agents:
            # a. 若 agent 已經在自己的 done 分支算過 loss / return，
            #    就直接拿 ag.last_episode_* ；這是最可靠的
            if ag.last_episode_step == 0:
                ag.last_episode_step = ag.episode_step          # 至少寫步數

            # b. timeout 但還沒算 loss 的情形
            if ag.last_episode_loss == 0.0:
                # 若有 update_actor_critic() 的結果，用那個
                if ag.episode_loss_history:
                    ag.last_episode_loss = ag.episode_loss_history[-1]
                else:
                    ag.last_episode_loss = float('nan')         # 讓平均值能顯示為 NaN

            if ag.last_episode_return == 0.0:
                ag.last_episode_return = ag.episode_reward

        # ========== 統計 & 重置 ==========
        steps   = [ag.last_episode_step for ag in self.agents]
        losses  = [ag.last_episode_loss for ag in self.agents]
        rets    = [ag.last_episode_return for ag in self.agents]
        dists   = [ag.last_episode_avg_dist for ag in self.agents]

        fleet_step = min(max(steps), MAX_STEP)
        fleet_loss   = float(np.mean(losses))
        fleet_return = float(np.mean(rets))
        fleet_dist   = float(np.mean(dists))

        self.logger.info(
            f"E{self.episode_id} ▶ step(last)={fleet_step} | "
            f"loss(avg)={fleet_loss:.4f} | return(avg)={fleet_return:.1f} | "
            f"avg_dist(avg)={fleet_dist:.2f} km"
        )
        
        # TensorBoard 記錄 (使用第一個 agent 的 stats_logger)
        if len(self.agents) > 0:
            self.agents[0].stats_logger.log_stat("fleet_episode_step", float(fleet_step), self.agents[0].total_steps)
            self.agents[0].stats_logger.log_stat("fleet_loss", float(fleet_loss), self.agents[0].total_steps)
            self.agents[0].stats_logger.log_stat("fleet_return", float(fleet_return), self.agents[0].total_steps)
        
        self.episode_id += 1

        # ---------- 真正 reset ----------
        reset_cmd = self._make_reset_script()      # ⇦ 換成固定起點版本
        full_cmd  = "\n".join(cmds) + "\n" + reset_cmd
        for ag in self.agents:
            ag.soft_reset()
        return full_cmd[:60000]          # Steam 通訊一次 64 k 限制 

    def _make_reset_script(self):
        lines = []
        for (side, name), (dbid, lat, lon) in self._init_pos.items():
            lines += [
                delete_unit(side=side, unit_name=name),
                add_unit(type='Ship', unitname=name, dbid=dbid,
                         side=side, Lat=lat, Lon=lon)
            ]
            if side == self.enemy_side:
                lines.append(set_unit_to_mission(unit_name=name,
                                                 mission_name='5VS5'))
        return "\n".join(lines)

    def get_reset_cmd(self, features):
        """
        收集 5 條 MyAgent 的重置 Lua，併成一段回傳給 run_loop。
        run_loop 只在『開新局』時呼叫一次。
        """
        cmd_list = []
        for ag in self.agents:
            cmd = ag.get_reset_cmd(features)
            if cmd:                          # 忽略 None 或空白
                cmd_list.append(cmd)

        joined = "\n".join(cmd_list)
        self.reset_cmd = joined       # ★★★ 關鍵：存給 run_loop 用
        return joined 