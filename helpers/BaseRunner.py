import logging
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import Constants
from utils.Metrics import Metrics
from utils.Optim import ScheduledOptim


class BaseRunner(object):
    def __init__(self, args):
        self.patience = 10  # The runner will stop when the validation score does not improve for 10 epochs

    def run(self, model, train_data, valid_data, test_data, args):
        loss_func = nn.CrossEntropyLoss(reduction='sum', ignore_index=Constants.PAD)
        params = model.parameters()
        adam = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-09)
        optimizer = ScheduledOptim(adam, args.d_model, args.n_warmup_steps)

        # ========= Move model to device =========#
        model = model.to(args.device)
        loss_func = loss_func.to(args.device)

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
            if current_validation_score > validation_history:
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

        # 训练完成或提前终止后，用测试集进行最终评估
        logging.info('\n  - (Final Test)')
        test_scores = self.test_epoch(model, test_data)

        logging.info(f"{'Metric':<15} {'Score':<20}")
        logging.info('-' * 35)  # 分割线
        for metric, score in test_scores.items():
            logging.info(f"{metric:<15} {score:<20.6f}")

        logging.info("\n- (Finished!!) \nBest validation scores:")
        logging.info(f"{'Metric':<15} {'Score':<20}")
        logging.info('-' * 35)  # 分割线
        for metric, score in best_scores.items():
            logging.info(f"{metric:<15} {score:<20.6f}")


    def train_epoch(self, model, training_data, loss_func, optimizer):
        # 设置模型为训练模式
        model.train()

        # 初始化总损失、总用户和总正确预测数
        total_loss = 0.0
        n_total_users = 0.0
        n_total_correct = 0.0

        # 使用 tqdm 包装 training_data 以显示训练进度
        for batch in tqdm(training_data, desc="Training Epoch", ncols=100):
            # 将批次中的数据移动到 GPU（如果可用）
            tgt, tgt_timestamp, tgt_idx = (item.cuda() for item in batch)

            # gold 是目标序列，去掉了序列的第一个元素
            gold = tgt[:, 1:]

            # 计算当前批次中非填充的用户数量
            n_users = gold.data.ne(Constants.PAD).sum().float()
            n_total_users += n_users  # 更新总用户数

            # 清空优化器的梯度
            optimizer.zero_grad()

            # 使用模型进行预测
            pred = model(tgt, tgt_timestamp, tgt_idx)

            # 计算损失和正确预测的数量
            loss, n_correct = self.get_performance(loss_func, pred, gold)
            # 反向传播以计算梯度
            loss.backward()

            # 更新模型参数
            optimizer.step()
            # 更新学习率（如果优化器支持的话）
            optimizer.update_learning_rate()

            # 累加正确预测的数量和总损失
            n_total_correct += n_correct
            total_loss += loss.item()

        # 返回平均损失和准确率
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
            for batch in tqdm(validation_data, desc="Testing Epoch"):
                # prepare data
                tgt, tgt_timestamp, tgt_idx = batch
                y_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()

                # forward
                pred = model(tgt, tgt_timestamp, tgt_idx)
                y_pred = pred.detach().cpu().numpy()

                metric = Metrics()
                scores_batch, scores_len = metric.compute_metric(y_pred, y_gold, k_list)
                n_total_words += scores_len
                for k in k_list:
                    scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                    scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

        for k in k_list:
            scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
            scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

        return scores

    def get_performance(self, crit, pred, gold):
        # gold.contiguous().view(-1) : [bth, max_len-1] -> [bth * (max_len-1)]
        loss = crit(pred, gold.contiguous().view(-1))
        # 获取 pred 中每行的最大值的索引，表示模型认为每个时间步最可能的类别,pred.max(1) 返回一个包含最大值和索引的元组，而 [1] 表示取出索引部分。
        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)  # 将 gold 转换为一维数组，确保它与 pred 的展平形状一致。
        n_correct = pred.data.eq(gold.data)  # 比较 pred 和 gold，返回一个布尔数组，表示每个位置是否预测正确。
        # gold.ne(Constants.PAD): 生成一个布尔数组，标记 gold 中不是填充符的部分。
        # masked_select(...): 只选择有效的（非填充）预测，确保不会将填充位置计入正确预测。
        # sum().float(): 最后计算正确预测的数量，并将其转换为浮点数。
        n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
        return loss, n_correct
