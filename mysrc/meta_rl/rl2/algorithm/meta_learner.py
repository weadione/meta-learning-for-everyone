import datetime
import os
import time
import warnings
from collections import deque
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

import numpy as np
import torch
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from torch.utils.tensorboard import SummaryWriter
from meta_rl.rl2.algorithm.buffer import Buffer
from meta_rl.rl2.algorithm.ppo import PPO
from meta_rl.rl2.algorithm.sampler import Sampler

class MetaLearner:
    def __init__(
        self,
        env: HalfCheetahEnv,
        env_name: str,
        agent: PPO,
        trans_dim: int,
        action_dim: int,
        hidden_dim: int,
        train_tasks: List[int],
        test_tasks: List[int],
        save_exp_name: str,
        save_file_name:str,
        load_exp_name: str,
        load_file_name: str,
        load_ckpt_num: str,
        device: torch.device,
        **config,
    ) -> None:
        self.env = env
        self.env_name = env_name
        self.agent = agent
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks

        self.num_iterations: int = config["num_iterations"]
        self.meta_batch_size: int = config["meta_batch_size"]
        self.num_samples: int = config["num_samples"]

        self.batch_size: int = self.meta_batch_size * config["num_samples"]
        self.max_step: int = config["max_step"]

        self.sampler = Sampler(
            env=env,
            agent=agent,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_step=config["max_step"],
        )

        self.buffer = Buffer(
            trans_dim=trans_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_size=self.batch_size,
            device=device,
        )

        if not save_file_name:
            save_file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.result_path = os.path.join("results", save_exp_name, save_file_name)
        self.writer = SummaryWriter(log_dir=self.result_path)

        if load_exp_name and load_file_name:
            ckpt_path = os.path.join(
                "results",
                load_exp_name,
                load_file_name, 
                "checkpoint_" + str(load_ckpt_num)+".pt"
            )
            ckpt = torch.load(ckpt_path)

            self.agent.policy.load_state_dict(ckpt["policy"])
            self.agent.vf.load_state_dict(ckpt["vf"])
            self.buffer = ckpt["buffer"]

        #early stop 세팅
        self.dq: deque = deque(maxlen=config["num_stop_conditions"])
        self.num_stop_conditions: int = config["num_stop_conditions"]
        self.stop_goal: int = config["stop_goal"]
        self.is_early_stopping = False

    def meta_train(self) -> None:
        #메타 트레이닝
        total_start_time = time.time()
        for it in range(self.num_iterations):
            start_time = time.time()

            print(f"====== Iteration {it} / {self.num_iterations} ======")
            #meta-batch task에 대한 trajs 샘플링
            indicies = np.random.randint(len(self.train_tasks), size=self.meta_batch_size)
            for i, index in enumerate(indicies):
                self.env.reset_task(index)
                self.agent.policy.is_deterministic = False

                print(f"[{i + 1}/{self.meta_batch_size}] collecting samples")
                trajs: List[Dict[str, np.ndarray]] = self.sampler.obtain_samples(
                    max_samples=self.num_samples,
                )
                self.buffer.add_trajs(trajs)

            batch = self.buffer.sample_batch()

            # policy과 value를 PPO 알고리즘에서 학습
            print(f"Start the meta-gradient update of iteration {it}")
            log_values = self.agent.train_model(self.batch_size, batch)

            #meta-test task에서 학습 성능 평가
            self.meta_test(it, total_start_time, start_time, log_values)

            if self.is_early_stopping:
                print(
                    f"\n================================================== \n"
                    f"The last {self.num_stop_conditions} meta-testing results are {self.dq}. \n"
                    f"And early stopping condition is {self.is_early_stopping}. \n"
                    f"Therefore, meta-training is terminated.",
                )
                break

    def visualize_within_tensorboard(self, test_result: Dict[str, Any], iteration: int) -> None:
        #meta-training, meta-testing 결과를 텐서보드에 기록
        self.writer.add_scalar("test/return", test_result["return"], iteration)
        if self.env_name == "vel":
            self.writer.add_scalar("test/sum_run_cost", test_result["sum_run_cost"], iteration)
            for step in range(len(test_result["run_cost"])):
                self.writer.add_scalar(
                    "run_cost/iteration_" + str(iteration),
                    test_result["run_cost"][step],
                    step,
                )
        self.writer.add_scalar("train/total_loss", test_result["total_loss"], iteration)
        self.writer.add_scalar("train/policy_loss", test_result["policy_loss"], iteration)
        self.writer.add_scalar("train/vaule_loss", test_result["value_loss"], iteration)
        self.writer.add_scalar("time/total_time", test_result["total_time"], iteration)
        self.writer.add_scalar("time/time_per_iter", test_result["time_per_iter"], iteration)
    
    def meta_test(
        self,
        iteration: int,
        total_start_time: float,
        start_time: float,
        log_values: Dict[str, float],
    ) -> None:
        #메타 테스팅
        test_results = {}
        test_return: float = 0.0
        test_run_cost = np.zeros(self.max_step)

        for index in self.test_tasks:
            self.env.reset_task(index)
            self.agent.policy.is_deterministic = True

            trajs: List[Dict[str, np.ndarray]] = self.sampler.obtain_samples(max_samples=self.max_step)
            test_return += np.sum(trajs[0]["rewards"]).item()

            if self.env_name == "vel":
                for i in range(self.max_step):
                    test_run_cost[i] += trajs[0]["infos"][i]

        test_results["return"] = test_return / len(self.test_tasks)
        if self.env_name == "vel":
            test_results["run_cost"] = test_run_cost / len(self.test_tasks)
            test_results["sum_run_cost"] = np.sum(abs(test_results["run_cost"]))
        test_results["total_loss"] = log_values["total_loss"]
        test_results["policy_loss"] = log_values["policy_loss"]
        test_results["value_loss"] = log_values["value_loss"]
        test_results["total_time"] = time.time() - total_start_time
        test_results["time_per_iter"] = time.time() - start_time

        self.visualize_within_tensorboard(test_results, iteration)

        #early stop 조건 검사
        if self.env_name == "dir":
            self.dq.append(test_results["return"])
            if all(list(map((lambda x: x >= self.stop_goal), self.dq))):
                self.is_early_stopping = True
        elif self.env_name == "vel":
            self.dq.append(test_results["sum_run_cost"])
            if all(list(map((lambda x: x <= self.stop_goal), self.dq))):
                self.is_early_stopping = True
        
        #학습 모델 저장
        if self.is_early_stopping:
            ckpt_path = os.path.join(self.result_path, "checkpoint_" + str(iteration) + ".pt")
            torch.save(
                {
                    "policy": self.agent.policy.state_dict(),
                    "vf": self.agent.vf.state_dict(),
                    "buffer": self.buffer
                },
                ckpt_path,
            )