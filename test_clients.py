"""
Quick test script to verify all LLM clients are working correctly.
"""

from clients.openai_client import OpenAIClient
from clients.anthropic_client import AnthropicClient
from clients.deepseek_client import DeepSeekClient
from clients.google_client import GoogleClient


def test_client(client, model_name):
    """Test a single client with a simple prompt."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print('='*60)
    
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello from {}'".format(model_name)}
    ]
    
    try:
        response = client.generate(test_messages)
        print(f"✅ SUCCESS")
        print(f"Response: {response[:100]}...")
        print(f"Latency: {client.last_latency:.2f}s")
        print(f"Tokens: {client.last_input_tokens} in, {client.last_output_tokens} out")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("LLM Client Test Suite - Testing All 11 Models (High/Mid/Low)")
    print("="*60)
    
    results = {}
    
    # Test OpenAI (3 models: High/Mid/Low)
    print("\n[1/4] OpenAI Models")
    openai_models = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]
    for model in openai_models:
        client = OpenAIClient(model=model)
        results[model] = test_client(client, model)
    
    # Test Anthropic (3 models: High/Mid/Low)
    print("\n[2/4] Anthropic Models")
    anthropic_models = [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5"
    ]
    for model in anthropic_models:
        client = AnthropicClient(model=model)
        results[model] = test_client(client, model)
    
    # Test Google (3 models: High/Mid/Low)
    print("\n[3/4] Google Models")
    google_models = ["gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview"]
    for model in google_models:
        client = GoogleClient(model=model)
        results[model] = test_client(client, model)
    
    # Test DeepSeek (2 models: Reasoning/General)
    print("\n[4/4] DeepSeek Models")
    deepseek_models = ["deepseek-reasoner", "deepseek-chat"]
    for model in deepseek_models:
        client = DeepSeekClient(model=model)
        results[model] = test_client(client, model)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    for model, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {model}")
    
    if passed == total:
        print("\n🎉 All clients working correctly!")
    else:
        print(f"\n⚠️  {total - passed} client(s) failed")


if __name__ == "__main__":
    main()
