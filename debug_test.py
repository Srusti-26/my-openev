import sys
sys.path.insert(0, '.')
from app import analyze_email

email = """From: john.doe@example.com
Subject: Urgent - Unable to access account

Hi Support Team,

I have been trying to log into my account since yesterday, but I keep getting an Invalid Credentials error.
This is affecting my ability to process client orders, and I need urgent assistance.

Thanks,
John Doe
Customer ID: 458921"""

r = analyze_email(email, "Hard")
print("classification :", r[0])
print("action         :", r[1])
print("confidence     :", r[3])
print("reply          :", r[6][:200] if r[6] else "(EMPTY)")
print("reward         :", r[13])
