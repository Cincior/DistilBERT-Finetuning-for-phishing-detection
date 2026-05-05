import os
from config import SAVE_PATH
from model_utils import load_or_init_model
from data_preprocessing import prepare_data
from train import run_training
from predict import explain_prediction, explain_attention

SAMPLE_EMAILS = [
    ("phishing", "Dear Customer enron, We detected an unusual sign-in attempt on your account from a new device and "
     "location. For your security, your account has been temporarily limited. Please confirm your "
     "identity and restore full access by clicking the secure link below: Verify My Account Now → "
     "http://security-check-account.com/verify If you do not verify your account within 24 hours, "
     "your account will be permanently suspended. Thank you for your cooperation, Security Team"),
    ("phishing", "URGENT: Your PayPal account has been suspended. Please log in at http://secure-paypal-login.com to verify your identity immediately."),
    ("phishing", "Congratulations! You've been selected to receive a $1000 Walmart Gift Card. Click here to claim your reward!"),
    ("phishing", "Invoice INV-99283 is overdue. Please download the attached PDF to avoid late fees and legal action."),
    ("phishing", "Dear Customer, we noticed suspicious activity on your credit card. Confirm your details now to prevent card blocking: http://bit.ly/bank-secure-auth"),
    ("phishing", "I am a lawyer representing a deceased relative who left you $10.5M. Please reply with your bank details to initiate the transfer."),
    ("normal",   "Hi Laura how you doing. You paid for me last time so let me take you for a dinner today. cant wait!"),
    ("normal",   "Hi Claire, Thank you for your recent purchase. We are confirming that your order has been received and is being "
                 "processed. You will receive a tracking number once the shipment is sent out. Please let us know if you have any "
                 "questions. Best, Amy Johnson Customer Support Agent Online Store X"),
    ("normal",   "I hope this email finds you well. I am checking on the status of the invoice I sent last week. It is urgent. Have you "
                 "processed the payment? Best regards, Tom McNish Content Manager GOOGLE Company"),
    ("phishing",   "URGENT URGENT give me money"),
]

ATTENTION_EMAILS = [
    ("normal", "I hope this email finds you well. I am checking on the status of the invoice I sent last week. Have you "
    "processed the payment? Best regards, Tom McNish Content Manager GOOGLE Company"),
    ("phishing", "URGENT: Your PayPal account has been suspended. Please log in at http://secure-paypal-login.com to verify your identity immediately."),
]


def main():
    tokenizer, model = load_or_init_model()

    if not os.path.exists(SAVE_PATH):
        df_train, df_val, df_test = prepare_data("phishingEmail.csv")
        run_training(model, tokenizer, df_train, df_val, df_test)

    print("\n" + "=" * 50)
    print("LIME EXPLANATIONS")
    print("=" * 50)
    for label, email in SAMPLE_EMAILS:
        print(f"\n[Expected: {label.upper()}]")
        explain_prediction(email, model, tokenizer)

    print("\n" + "=" * 50)
    print("ATTENTION EXPLANATIONS")
    print("=" * 50)
    for label, email in ATTENTION_EMAILS:
        print(f"\n[Expected: {label.upper()}]")
        explain_attention(email, model, tokenizer)


if __name__ == "__main__":
    main()
