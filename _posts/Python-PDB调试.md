---
layout:     post
title:      Python PDB调试
subtitle:   pycharm jupyter PDB调试
date:       2019-07-05
author:     BY
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - Python
---

```
python
import pdb 
a = "aaa"
pdb.set_trace() 
b = "bbb"
c = "ccc"
final = a + b + c 
print final
```

| 命令                | 解释                       |
| ------------------- | -------------------------- |
| break 或 b 设置断点 | 设置断点                   |
| continue 或 c       | 继续执行程序               |
| list 或 l           | 查看当前行的代码段         |
| step 或 s           | 进入函数                   |
| return 或 r         | 执行代码直到从当前函数返回 |
| exit 或 q           | 中止并退出                 |
| next 或 n           | 执行下一行                 |
| pp                  | 打印变量的值               |
| help                | 帮助                       |