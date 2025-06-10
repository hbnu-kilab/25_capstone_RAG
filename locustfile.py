from locust import HttpUser, task, between

class MyUser(HttpUser):
    wait_time = between(1, 2)  # 요청 간 대기 시간

    @task
    def load_home(self):
        self.client.get("/")  # 실제 요청