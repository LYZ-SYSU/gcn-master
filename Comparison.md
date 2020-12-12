- ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200707211932249.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mjk3MzY3OA==,size_16,color_FFFFFF,t_70)

- $$
  D^{-1/2}AD^{-1/2}HX
  $$
  
- Untrainable:
  
  - A=A.(F*F^T) untrainable  0.75600
  - fixed A (D adaptive) 0.52100
  - fixed D (A adaptive) 0.16100 (don't work)
  
- Trainable:

  - 

  | Accuracy    | Both A and D | Only A( Fixed D) | Only D (Fixed A) |
  | ----------- | ------------ | ---------------- | ---------------- |
  | Untrainable | 0.75600      | 0.16100          | 0.52100          |
  | Trainable   | ING          | \                |                  |

  