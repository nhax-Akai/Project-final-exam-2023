# Trình bày nghiên cứu câu 1 học máy
MSSV: 52000376 Họ tên: Phạm Phong Nhã
## 1) Tìm hiểu, so sánh các phương pháp Optimizer trong huấn luyện mô hình học máy
### SGD (Stochastic Gradient Descent)
SGD là một thuật toán tối ưu hoá lặp được sử dụng phổ biến trong học máy. Nó là một biến thể của gradient descent, thực hiện cập nhật tham số mô hình (trọng số) dựa trên gradient của hàm loss được tính toán trên một tập hợp con được chọn ngẫu nhiên từ dữ liệu huấn luyện, thay vì toàn bộ dữ liệu [1].  
Ý tưởng cơ bản của SGD là lấy mẫu một tập hợp con nhỏ ngẫu nhiên từ dữ liệu huấn luyện, được gọi là mini-batch, và tính toán gradient của hàm loss đối với các tham số mô hình chỉ sử dụng tập con đó. Gradient này sau đó được sử dụng để cập nhật các tham số [2]. Quá trình này được lặp lại với một mini-batch ngẫu nhiên mới cho đến khi thuật toán hội tụ hoặc đạt đến tiêu chí dừng được xác định trước.
#### SGD có một số ưu điểm so với gradient descent tiêu chuẩn, chẳng hạn như:
* Hội tụ nhanh hơn [3]: SGD có thể hội tụ nhanh hơn gradient descent, đặc biệt là khi dữ liệu huấn luyện lớn.
* Thoát khỏi các cực tiểu cục bộ [4]: SGD có khả năng thoát khỏi các cực tiểu cục bộ, điều này có thể giúp nó đạt được kết quả tốt hơn so với gradient descent tiêu chuẩn.
#### Tuy nhiên, SGD cũng có một số nhược điểm:
* Yêu cầu nhiều lần lặp hơn [5]: SGD thường yêu cầu nhiều lần lặp hơn gradient descent để hội tụ.
* Nhạy cảm với tốc độ học [5]: Tốc độ học cần được điều chỉnh cẩn thận để đảm bảo hội tụ của SGD.



ADAM
Adam (Adaptive Moment Estimation) là một thuật toán tối ưu hóa phổ biến trong huấn luyện mạng nơ-ron để cải thiện hiệu quả và tốc độ hội tụ [6].
Thuật toán kết hợp 2 ý tưởng chính: động lượng (momentum) và RMSProp. Adam duy trì trung bình trượt của gradient, tương ứng với giá trị trung bình và phương sai của chúng. Phần trung bình trượt gradient giúp Adam duy trì hướng di chuyển ngay cả khi gradient thay đổi nhỏ dần. Phần phương sai giúp Adam thích ứng tốc độ học riêng biệt cho từng tham số [7].
Ngoài ra, Adam còn áp dụng kỹ thuật chỉnh sửa sai số ban đầu của trung bình trượt để cải thiện hiệu năng [6].
Adam phổ biến nhờ khả năng hội tụ nhanh [8], xử lý tốt gradient thưa thớt & nhiễu [7], đồng thời không đòi hỏi nhiều siêu tham số như một số thuật toán khác [9].

RMSProp
RMSProp (Root Mean Square Propagation) là một thuật toán tối ưu hóa được sử dụng trong học máy và học sâu để tối ưu hóa quá trình huấn luyện mạng nơ-ron [10].
Không giống Adagrad tích lũy tất cả gradient, RMSProp tính trung bình di động của bình phương các gradient [7]. Điều này cho phép điều chỉnh tốc độ học một cách mượt mà hơn, và ngăn chặn tốc độ học giảm quá đột ngột.
RMSProp cũng sử dụng một yếu tố suy giảm để điều tiết ảnh hưởng của gradient quá khứ, cho phép gán trọng số lớn hơn cho gradient gần đây [11].
Một ưu điểm của RMSProp là khả năng xử lý tốt hàm mục tiêu thay đổi theo thời gian. Trong khi đó, trong trường hợp này Adagrad có thể hội tụ nhanh quá mức [12]. RMSProp có thể điều chỉnh tốc độ học phù hợp với sự thay đổi của hàm mục tiêu.


Adagrad
Adagrad (Adaptive Gradient) là một thuật toán tối ưu hoá được sử dụng để tối ưu hoá quá trình huấn luyện của các mạng nơ-ron.
Thuật toán Adagrad điều chỉnh tốc độ học của mỗi tham số của mạng nơ-ron một cách thích ứng trong quá trình huấn luyện. Cụ thể, nó tỉ lệ tốc độ học của mỗi tham số dựa trên các gradient lịch sử được tính toán cho tham số đó. Các tham số có gradient lớn được cho tốc độ học nhỏ hơn, trong khi những tham số có gradient nhỏ được cho tốc độ học lớn hơn. 
Adagrad (Adaptive Gradient) là một thuật toán tối ưu hoá được sử dụng để tối ưu hoá quá trình huấn luyện của các mạng nơ-ron [13].
Thuật toán Adagrad điều chỉnh tốc độ học của mỗi tham số của mạng nơ-ron một cách thích ứng trong quá trình huấn luyện. Cụ thể, nó tỉ lệ tốc độ học của mỗi tham số dựa trên các gradient lịch sử được tính toán cho tham số đó [5]. Các tham số có gradient lớn được cho tốc độ học nhỏ hơn, trong khi những tham số có gradient nhỏ được cho tốc độ học lớn hơn.

Các điểm nổi bật của Adagrad:
Tốc độ học thích ứng [5]: Adagrad điều chỉnh tốc độ học cho từng tham số dựa trên lịch sử gradient của nó, giúp tránh giảm tốc quá nhanh và cho phép hội tụ nhanh hơn.
Thích hợp cho dữ liệu thưa thớt [14]: Adagrad xử lý tốt dữ liệu thưa thớt bằng cách gán tốc độ học lớn hơn cho các tham số có gradient nhỏ.

Một số nhược điểm của Adagrad:
Tính toán tốn kém [15]: Adagrad cần lưu trữ lịch sử gradient của tất cả các tham số, có thể tốn kém về bộ nhớ và tính toán.
Giảm hiệu quả sau khi hội tụ [15]: Sau khi hội tụ, Adagrad có thể tiếp tục giảm tốc độ học, dẫn đến việc huấn luyện bị chậm lại.

Optimizer	Ưu điểm	Nhược điểm
SGD	- Dễ dàng thực hiện và tính toán hiệu quả. 
- Hiệu quả đối với các tập dữ liệu lớn với không gian đặc trưng nhiều chiều.	- SGD có thể bị mắc kẹt trong các cực tiểu cục bộ.
- Độ nhạy cao với tốc độ học ban đầu
Adam	- Hiệu quả và dễ thực hiện. 
- Áp dụng cho các tập dữ liệu lớn và mô hình nhiều chiều. 
- Khả năng khái quát hóa tốt.	Cần phải điều chỉnh cẩn thận các hyperparameter. 
RMSProp	- Tốc độ học thích ứng trên mỗi tham số giúp hạn chế sự tích lũy độ dốc. 
- Hiệu quả đối với các mục tiêu không cố định.	- Có thể có tốc độ hội tụ chậm trong một số trường hợp.
Adagrad	- Tỷ lệ học tập thích ứng cho mỗi tham số. 
- Hiệu quả đối với dữ liệu thưa thớt.	- Việc tích lũy gradient bình phương trong mẫu số có thể khiến tốc độ học giảm xuống quá nhanh. 
- Có thể dừng việc học quá sớm.


2)Tìm hiểu về Continual Learning và Test Production khi xây dựng một giải pháp học máy để giải quyết một bài toán nào đó.

Continual Learning
Khái niệm:

Continual Learning(Học liên tục) là một quá trình trong đó một mô hình học từ các luồng dữ liệu mới mà không cần phải đào tạo lại. [17]
Nó trái ngược với các phương pháp tiếp cận truyền thống, trong đó mô hình được đào tạo trên một tập dữ liệu cố định, được triển khai và đạo tạo lại định kỳ, các mô hình học liên tục cập nhật lặp đi lặp lại các tham số của chúng để phản ánh các phân phối mới trong dữ liệu. [17]
Trong quá trình sau, mô hình cải thiện chính nó bằng cách học từ lần lặp mới nhất và cập nhật kiến thức khi có dữ liệu mới. Vòng đời mô hình học máy liên tục cho phép các mô hình duy trình tính liên quan theo thời gian do chất lượng động của chúng.
Quá trình của Continual Learning[16]:
Bước 1: Initial training - huấn luyện mô hình trên tập dữ liệu ban đầu. Mô hình học một tập các tham số khởi đầu dựa trên các mẫu mà nó nhận thấy trong dữ liệu.
Bước 2: Deployment - mô hình được sử dụng để thực hiện nhiệm vụ dự định. Trong thời gian này, dữ liệu mới liên quan đến nhiệm vụ và môi trường được thu thập.
Bước 3: Data rehearsal - mô hình được điều chỉnh thường xuyên bằng cách ôn lại các kinh nghiệm trước đó, để không quên các thông tin đã học trước đó, trong khi được huấn luyện bằng dữ liệu mới.
Bước 4: Continuous learning strategy - một chiến lược học tập liên tục được áp dụng để thích ứng và cải thiện hiệu suất mô hình.
Bước 5: Revaluation and monitoring - hiệu suất mô hình được đánh giá lặp đi lặp lại về độ chính xác, khả thi, hành vi thực tế và độ chệch.



Ưu điểm:
Khả năng khái quát hóa và dự đoán tốt hơn nhờ tích lũy kiến thức theo thời gian.
Giữ lại và xây dựng dựa trên kiến thức đã học.
Thích ứng tốt với dữ liệu và kiến thức mới.
Hạn chế:
Khó quản lý các các phiên bản mô hình khác nhau
Cần xử lý liên tục dữ liệu mới, dễ bị ảnh hưởng bởi dữ liệu trôi dạt


Test Production

Khái niệm: Test Production là quá trình kiểm tra đánh giá mô hình học máy sau khi đã huấn luyện xong trên tập dữ liệu thực tế. Mục đích là xác định xem mô hình đó có thể hoạt động tốt và đáp ứng yêu cầu nghiệp vụ hay không khi triển khai trong môi trường thực tế.
Các bước chính trong Test Production bao gồm:
1.Chuẩn bị tập dữ liệu kiểm tra (test set): Tập dữ liệu này cần có cùng đặc tính phân phối với dữ liệu huấn luyện và độc lập hoàn toàn với dữ liệu đã dùng để huấn luyện mô hình.
2.Triển khai mô hình: Triển khai mô hình đã huấn luyện vào môi trường thực tế giống hệt với môi trường production.
3.Chạy thử nghiệm: Cho mô hình dự đoán trên tập dữ liệu kiểm tra và thu thập các metric (accuracy, recall, precision, F1 score,...).
4.Phân tích và đánh giá: So sánh các metric thu được với ngưỡng mong đợi và yêu cầu thực tế để đánh giá hiệu quả của mô hình.
5.Tinh chỉnh và cải tiến: Nếu cần thiết có thể quay lại cải tiến quy trình xây dựng và huấn luyện mô hình để nâng cao hiệu quả trước khi triển khai chính thức.




TÀI LIỆU THAM KHẢO
[1] L. Bottou, “Large-Scale Machine Learning with Stochastic Gradient Descent,” in COMPSTAT, 2010, pp. 177–186.
[2] Y. LeCun, L. Bottou, G. B. Orr, and K. Müller, “Efficient backprop,” in Neural Networks: Tricks of the Trade, 2012, pp. 9–48.
[3] N. Qian, “On the momentum term in gradient descent learning algorithms,” Neural Networks, vol. 12, no. 1, pp. 145–151, 1999.
[4] L. Bottou, “Stochastic gradient tricks,” in Neural Networks: Tricks of the Trade, 2012, pp. 430–445.
[5] S. Ruder, “An overview of gradient descent optimization algorithms,” CoRR, vol. abs/1609.04747, 2016.
[6] D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” in ICLR, 2015.
[7] S. Ruder, “An overview of gradient descent optimization algorithms,” arXiv, 2016
[8] J. Dozat, “Incorporating Nesterov Momentum into Adam,” in ICLR Workshop, 2016.
[9] L. Luo, Y. Xiong, Y. Liu, and X. Sun, “Adaptive Gradient Methods with Dynamic Bound of Learning Rate,” in ICLR, 2019.
[10] G. Hinton et al., "Lecture 6a Overview of mini-batch gradient descent," Coursera Lecture slides, 2012.
[11] L. Bottou et al., "Optimization Methods for Large-Scale Machine Learning," SIAM Review, vol. 60, no. 2, pp. 223-311, 2018.
[12] J. Schmidhuber, “Deep Learning in Neural Networks: An Overview,” Neural Networks, vol. 61, pp. 85-117, 2015.
[13] Duchi et al., "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization," JMLR, 2011.
[14] Zeiler, "ADADELTA: An Adaptive Learning Rate Method," arXiv:1212.5701, 2012.
[15] Wilson et al., "The Marginal Value of Adaptive Gradient Methods in Machine Learning," NIPS, 2017.
[16] F. M. Castro et al., “Continual learning in practice,” IEEE Signal Processing Magazine, vol. 38, no. 6, pp. 101–114, 2021.
[17] A. S. Polyak and L. Wolf, “Channel-wise cradle for lifelong learning,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 1297–1306.
