# BaseRunner.py

import logging
import time
import os  # Added for os.path.exists in case it was missing, but seems present in your original too.

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import Constants
from utils.Metrics import Metrics
from utils.Optim import ScheduledOptim


class BaseRunner(object):
    def __init__(self, args):  # args is already the parameter here
        self.patience = 10  # The runner will stop when the validation score does not improve for 10 epochs
        self.device = args.device  # ADDED: Store the device from args for internal use

    def run(self, model, train_data, valid_data, test_data, args):
        loss_func = nn.CrossEntropyLoss(reduction='sum', ignore_index=Constants.PAD)
        params = model.parameters()
        adam = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-09)
        optimizer = ScheduledOptim(adam, args.d_model, args.n_warmup_steps)

        # ========= Move model to device =========#
        model = model.to(self.device)  # MODIFIED: Move model to the configured device
        loss_func = loss_func.to(self.device)  # MODIFIED: Move loss function to the configured device

        validation_history = 0.0
        best_scores = {}
        epochs_without_improvement = 0  # 用于记录没有提升的epoch数量

        for epoch_i in range(args.epoch):
            logging.info(f'\n[ Epoch {epoch_i} ]')
            start = time.time()

            # 训练模型
            train_loss, train_accu = self.train_epoch(model, train_data, loss_func, optimizer)
            logging.info(
                f'  \n- (Training)   '
                f'Loss: {train_loss:8.5f} | '
                f'Accuracy: {100 * train_accu:3.3f}% | '
                f'Elapsed: {(time.time() - start) / 60:3.3f} min')

            # 每个 epoch 结束后，用验证集进行评估
            start = time.time()
            validation_scores = self.test_epoch(model, valid_data)
            logging.info('\n  - (Validation)')

            logging.info(f"{'Metric':<15} {'Score':<20}")
            logging.info('-' * 35)  # 分割线
            for metric, score in validation_scores.items():
                logging.info(f"{metric:<15} {score:<20.6f}")

            logging.info(f'Validation use time: {(time.time() - start) / 60:.3f} min')

            # 检查验证集上的分数是否有提升
            current_validation_score = sum(validation_scores.values())
            if not best_scores or current_validation_score > validation_history:  # MODIFIED: Check if best_scores is empty for first epoch
                # 如果有提升，则保存模型并重置无提升计数器
                logging.info(f"\nBest Validation at Epoch: {epoch_i}, model has been saved.")
                validation_history = current_validation_score
                best_scores = validation_scores
                torch.save(model.state_dict(), args.model_path)
                epochs_without_improvement = 0  # 重置计数器
            else:
                # 如果没有提升，则增加无提升计数器
                epochs_without_improvement += 1
                logging.info(f"No improvement in validation score for {epochs_without_improvement} epochs.")

            # 提前终止条件检查
            if epochs_without_improvement >= self.patience:
                logging.info(f"\nEarly stopping triggered after {epoch_i + 1} epochs.")
                break

        # Load best model for final test
        if os.path.exists(args.model_path):
            logging.info(f"Loading best model from {args.model_path} for final testing.")
            model.load_state_dict(
                torch.load(args.model_path, map_location=self.device))  # MODIFIED: Use self.device for map_location
        else:
            logging.warning("No saved model found. Testing with the last model state.")

        # 训练完成或提前终止后，用测试集进行最终评估
        logging.info('\n  - (Final Test)')
        test_scores = self.test_epoch(model, test_data)

        logging.info(f"{'Metric':<15} {'Score':<20}")
        logging.info('-' * 35)  # 分割线
        for metric, score in test_scores.items():
            logging.info(f"{metric:<15} {score:<20.6f}")

        logging.info("\n- (Finished!!) \nBest validation scores:")
        if best_scores:  # ADDED: Check if best_scores is populated
            logging.info(f"{'Metric':<15} {'Score':<20}")
            logging.info('-' * 35)  # 分割线
            for metric, score in best_scores.items():
                logging.info(f"{metric:<15} {score:<20.6f}")
        else:
            logging.info("No best validation scores recorded.")

    def train_epoch(self, model, training_data, loss_func, optimizer):
        # 设置模型为训练模式
        model.train()

        # 初始化总损失、总用户和总正确预测数
        total_loss = 0.0
        n_total_users = 0.0
        n_total_correct = 0.0

        # 使用 tqdm 包装 training_data 以显示训练进度
        for batch in tqdm(training_data, desc="Training Epoch", ncols=100):
            # 将批次中的数据移动到 GPU（如果可用）-> MODIFIED: Move to self.device
            history_seq, history_seq_timestamp, history_seq_idx = (item.to(self.device) for item in
                                                                   batch)  # MODIFIED to use self.device

            # gold 是真实标签，取出 history_seq 从第 1 个位置开始的所有元素，省略第 0 个元素
            gold = history_seq[:, 1:]

            # 计算当前批次中非填充的用户数量
            n_users = gold.data.ne(Constants.PAD).sum().float()

            # ADDED: Skip batch if no valid users to prevent division by zero later (original code might implicitly handle this or error)
            if n_users.item() == 0:
                continue

            n_total_users += n_users  # 更新总用户数

            # 清空优化器的梯度
            optimizer.zero_grad()

            # 使用模型进行预测
            # input_seq = history_seq[:, :-1]
            # input_seq_timestamp = history_seq_timestamp[:, :-1]
            # pred = model(input_seq, input_seq_timestamp, history_seq_idx)

            # 计算损失和正确预测的数量
            # loss_func is already on self.device. history_seq, etc., are now on self.device. gold is derived from history_seq.
            loss, n_correct = model.get_performance(history_seq,
                                                    history_seq_timestamp,
                                                    history_seq_idx,
                                                    loss_func,
                                                    gold)
            # 反向传播以计算梯度
            loss.backward()

            # 更新模型参数
            optimizer.step()
            # 更新学习率（如果优化器支持的话）
            optimizer.update_learning_rate()

            # 累加正确预测的数量和总损失
            n_total_correct += n_correct  # n_correct should be a scalar tensor or float from model.get_performance
            total_loss += loss.item()

        # 返回平均损失和准确率
        if n_total_users.item() == 0:  # ADDED: Handle case where no users were processed in any batch
            return 0.0, 0.0
        return total_loss / n_total_users, n_total_correct / n_total_users

    def test_epoch(self, model, validation_data, k_list=[10, 20, 50, 100]):
        """ Epoch operation in evaluation phase """
        model.eval()

        scores = {}
        for k in k_list:
            scores['hits@' + str(k)] = 0
            scores['map@' + str(k)] = 0

        n_total_words = 0
        with torch.no_grad():
            for batch in tqdm(validation_data,
                              desc="Evaluating Epoch"):  # Changed desc for clarity from "Testing Epoch"
                # 将批次中的数据移动到 GPU（如果可用）-> MODIFIED: Move to self.device
                history_seq, history_seq_timestamp, history_seq_idx = (item.to(self.device) for item in
                                                                       batch)  # MODIFIED to use self.device

                # gold 是真实标签，取出 history_seq 从第 1 个位置开始的所有元素，省略第 0 个元素
                # gold_on_device will be on self.device
                gold_on_device = history_seq[:, 1:].contiguous().view(-1)

                # forward
                # pred_on_device will be on self.device
                pred_on_device = model(history_seq, history_seq_timestamp, history_seq_idx)

                # Original logic moves to CPU for numpy operations, which is fine
                y_pred_cpu = pred_on_device.detach().cpu().numpy()
                gold_cpu = gold_on_device.detach().cpu().numpy()

                metric = Metrics()
                # compute_metric expects numpy arrays
                scores_batch, scores_len = metric.compute_metric(y_pred_cpu, gold_cpu, k_list)

                # ADDED: Only update scores if scores_len > 0 (i.e., valid samples were processed)
                if scores_len > 0:
                    n_total_words += scores_len
                    for k in k_list:
                        scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                        scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

        if n_total_words > 0:
            for k in k_list:
                scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
                scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words
        else:  # ADDED: Handle case where n_total_words is 0 to avoid division by zero and return 0 scores
            for k in k_list:
                scores['hits@' + str(k)] = 0.0
                scores['map@' + str(k)] = 0.0

        return scores

    def get_performance(self, crit, pred, gold):
        # gold.contiguous().view(-1) : [bth, max_len-1] -> [bth * (max_len-1)]
        # crit (loss_func) is on self.device
        # pred (model output) is on self.device
        # gold (target sequence) is on self.device
        # No device-specific changes needed here as inputs are expected to be on the correct device.
        loss = crit(pred, gold.contiguous().view(-1))
        # 获取 pred 中每行的最大值的索引，表示模型认为每个时间步最可能的类别,pred.max(1) 返回一个包含最大值和索引的元组，而 [1] 表示取出索引部分。
        pred_choice = pred.max(1)[1]  # pred_choice will be on the same device as pred
        gold_flat = gold.contiguous().view(-1)  # 将 gold 转换为一维数组，确保它与 pred 的展平形状一致。
        n_correct = pred_choice.data.eq(gold_flat.data)  # 比较 pred 和 gold，返回一个布尔数组，表示每个位置是否预测正确。
        # gold.ne(Constants.PAD): 生成一个布尔数组，标记 gold 中不是填充符的部分。
        mask = gold_flat.ne(Constants.PAD).data
        # masked_select(...): 只选择有效的（非填充）预测，确保不会将填充位置计入正确预测。
        # sum().float(): 最后计算正确预测的数量，并将其转换为浮点数。
        n_correct = n_correct.masked_select(mask).sum().float()
        return loss, n_correct
