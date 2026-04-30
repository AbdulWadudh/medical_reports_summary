import json
import logging
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NHAclient:
    TOKEN_URL = "https://aaehackathon.nhaad.in/controlplane/iudx/v2/auth/token"
    COMPLETIONS_URL = "https://aaehackathon.nhaad.in/nhm-proxy/v1/chat/completions"

    def __init__(self, client_id: str, client_secret: str):
        self.id = client_id
        self.secret = client_secret
        self.token = self._get_token()

    def _get_token(self) -> str:
        logger.info("Fetching new auth token...")
        resp = httpx.post(
            self.TOKEN_URL,
            headers={
                "clientId": self.id,
                "clientSecret": self.secret
            },
            follow_redirects=True
        )
        resp.raise_for_status()
        data = resp.json()
        
        result = data.get("result", {})
        token = result.get("access_token") or result.get("token")
        if not token:
            raise ValueError(f"Token not found in response: {data}")
        
        logger.info("Auth token obtained successfully")
        return token

    def completion(self, model: str, messages: list, **kwargs) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "auth_token": self.token,
            **kwargs
        }

        with httpx.Client(timeout=None) as client:
            resp = client.post(self.COMPLETIONS_URL, json=payload)
            
            if self._is_token_expired(resp):
                logger.info("Token expired, refreshing token and retrying...")
                self.token = self._get_token()
                payload["auth_token"] = self.token
                resp = client.post(self.COMPLETIONS_URL, json=payload)
            
            return resp.json()

    def _is_token_expired(self, resp: httpx.Response) -> bool:
        if resp.status_code == 401:
            return True
        
        if resp.status_code == 200:
            try:
                data = resp.json()
                if isinstance(data, dict):
                    error = data.get("error", {})
                    if isinstance(error, dict):
                        message = error.get("message", "")
                        if "expired" in message.lower() or "token" in message.lower():
                            return True
            except Exception:
                pass
        
        return False


def main():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    client = NHAclient(
        client_id=os.environ["NHA_CLIENT_ID"],
        client_secret=os.environ["NHA_CLIENT_SECRET"],
    )

    resp = client.completion(
        model="Ministral 8B",
        messages=[{"role": "user", "content": "Say hello in 3 words"}],
        metadata={"problem_statement": 2},
    )
    print(resp)


if __name__ == "__main__":
    main()
