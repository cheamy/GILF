def min_jumps_to_reach_end(n, teleports=[]):
    """
    计算小红在长度为n的数组上，利用弹力鞋和单向传送阵，至少需要弹跳多少次才能到达最后一个元素。

    :param n: 数组的长度
    :param teleports: 传送阵的位置列表，默认为空列表
    :return: 至少需要弹跳的次数
    """
    jumps = 0  # 记录弹跳次数
    current_position = 0  # 小红的当前位置
    jump_distance = 1  # 弹力鞋的当前弹跳距离

    teleports = sorted(set(teleports))  # 对传送阵位置进行去重并排序
    teleport_index = 0  # 当前考虑的传送阵索引

    while current_position < n - 1:  # 当小红没有到达最后一个元素时
        if teleports and teleport_index < len(teleports) and teleports[
            teleport_index] <= current_position + jump_distance:
            # 如果当前传送阵在弹跳范围内，则使用传送阵
            current_position = teleports[teleport_index]
            jump_distance = 1  # 重置弹跳距离
            teleport_index += 1  # 考虑下一个传送阵
        else:
            # 否则，使用弹力鞋弹跳
            current_position += jump_distance
            jump_distance *= 2  # 弹跳距离翻倍
            # 如果当前位置超过了最后一个传送阵的位置，则不再考虑传送阵
            if teleports and current_position > teleports[-1]:
                teleports = []
        jumps += 1  # 弹跳次数加1

    return jumps


# 示例用法：
n = 6
teleports = [3, 2, -1, -1, 6, -1]  # 传送阵的位置
print(min_jumps_to_reach_end(n, teleports))  # 输出最少弹跳次数