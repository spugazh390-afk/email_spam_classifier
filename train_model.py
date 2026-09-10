import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from model import SpamClassifier

# Rich training dataset of realistic Spam and Ham emails
TRAINING_DATA = [
    # --- SPAM EMAILS ---
    ("URGENT: Your account access has been suspended! Verify your password and login credentials immediately to avoid permanent deletion. Click here to confirm identity.", 1),
    ("Congratulations! You have been selected as the official winner of our $1,000,000 cash lottery prize! Reply with your bank details and claim your reward today.", 1),
    ("Exclusive Offer: Guaranteed 500% profit with Bitcoin crypto trading robot. Deposit $250 now and become rich overnight. Limited slots available, click now!", 1),
    ("Invoice Overdue: Unpaid outstanding balance of $3,450.00 for services rendered. Please find attached malicious document or click link to remit payment immediately.", 1),
    ("Claim your FREE $500 Walmart Gift Card! You were chosen in our promotional customer sweepstakes. Complete our 1-minute survey to receive voucher now.", 1),
    ("ATTENTION: Pre-approved instant cash loan up to $50,000 with 0% interest! No credit check required, bad credit ok. Apply online right now.", 1),
    ("Buy cheap generic medications online without prescription! 80% discount on Viagra, Cialis, weight loss pills. Fast discreet international shipping.", 1),
    ("Security Alert: Unauthorized sign-in detected on your Microsoft Office 365 account from Russia. Click here to verify your identity and reset password.", 1),
    ("Earn $5,000 weekly working from home! No experience required, just 2 hours a day. Click the link to register your starter kit and start earning money.", 1),
    ("Final Notice: Your vehicle warranty is about to expire! Call 1-800-FAKE-NUM immediately to renew your coverage before penalties apply.", 1),
    ("Dear Beneficiary, We are releasing $10.5 Million USD from Central Bank. Kindly send your full name, passport copy, and fee of $150 to process wire transfer.", 1),
    ("Hot Singles in your area are waiting to chat! Click here to view unread private photos and connect with verified profiles near you tonight.", 1),
    ("You have won an iPhone 16 Pro! Only pay $1 shipping fee to claim your brand new smartphone. Enter credit card details now before stock runs out.", 1),
    ("Action Required: Your Netflix subscription failed to renew. Update your billing credit card information immediately or your streaming service will be terminated.", 1),
    ("Secret weight loss discovery! Doctors hate this simple trick that burns 20 lbs of fat in 7 days. Watch free presentation before it gets banned.", 1),
    ("PayPal Alert: We noticed unusual activity on your PayPal wallet. Confirm your social security number and debit card to restore your account limits.", 1),
    ("Risk-Free Casino Bonus: 200 Free Spins + 300% deposit match bonus waiting in your account. Play slots, roulette and cash out huge jackpots instantly!", 1),
    ("IRS Tax Refund Notice: You have an unclaimed tax refund payment of $1,842.50. Click our secure link to submit your direct deposit banking details.", 1),
    ("Double your money in 48 hours! Join our private Telegram insider trading group with 99.8% accurate forex signals. Register free today.", 1),
    ("Urgent wire transfer request: I am in an emergency meeting and need you to purchase 10 Apple gift cards and email the redemption codes right away.", 1),
    ("Package Delivery Failed: DHL Express could not deliver your parcel. Download the attached shipping receipt and reschedule delivery within 24 hours.", 1),
    ("Get ranked #1 on Google in 14 days! Our automated SEO software will blast 100,000 backlinks to your website. Order now for 90% discount.", 1),
    ("Your Apple ID has been locked for security reasons. Click here to confirm your billing address and unlock your devices.", 1),
    ("Special promotion: Lowest price guaranteed for Rolex replica watches. Free worldwide shipping for orders placed in the next 3 hours.", 1),
    ("Urgent: Amazon customer order cancellation warning. A purchase of $999 MacBook was made on your card. Call support if this was not you.", 1),
    ("You are eligible for debt relief forgiveness! Eliminate up to 80% of your credit card debt legally under new government programs. Call today.", 1),
    ("Exclusive VIP invitation to join billionaire investment syndicate. Guaranteed passive income streams. Limited time offer, subscribe now.", 1),
    ("Free trial subscription! Try our male enhancement supplements completely free of charge. Just pay shipping and handling.", 1),
    ("Security Alert: Your cloud backup has expired. All personal photos and documents will be permanently purged in 12 hours unless you renew.", 1),
    ("Notice of unclaimed inheritance: Attorney Barrister seeking next of kin for deceased estate of $8.2M. Contact urgently for mutual benefit.", 1),

    # --- HAM (LEGITIMATE) EMAILS ---
    ("Hi team, please find attached the agenda for tomorrow's sprint retrospective meeting at 10:00 AM. Let me know if you have any discussion items to add.", 0),
    ("Thank you for your order! Your payment for order #84920 has been received and your items are being prepared for shipment via FedEx tracking.", 0),
    ("Hi Sarah, are you free for a quick sync this afternoon regarding the Q3 roadmap presentation? Let me know what time works best for your schedule.", 0),
    ("Weekly Engineering Digest: Updates on our microservice migration, Python 3.12 performance benchmarks, and scheduled system maintenance this Saturday.", 0),
    ("Good morning, could you please review the attached draft proposal for the marketing campaign before we present it to the client on Thursday?", 0),
    ("GitHub Notification: Pull request #142 'Refactor authentication middleware' has been approved and merged into the development branch by Alex.", 0),
    ("Your monthly electricity bill statement for August is now available to view on your utility account portal. Due date is September 25th.", 0),
    ("Hi Dad, hope you're having a great week! Let me know if you and mom want to meet up for Sunday lunch at the usual cafe.", 0),
    ("Campus Seminar Announcement: The Department of Computer Science invites all faculty and students to a guest lecture on Natural Language Processing this Friday.", 0),
    ("Customer Support Ticket #58291: Your inquiry regarding dashboard analytics export has been resolved. Please rate our support response.", 0),
    ("Reminder: Project status report submission is due by 5:00 PM today. Please update your team's Jira board tickets accordingly.", 0),
    ("Hi everyone, please note that the office Wi-Fi network will undergo routine maintenance tonight between 11 PM and 1 AM. Downtime should be minimal.", 0),
    ("Here are the meeting minutes and action items from our quarterly executive strategy review. Thanks everyone for your active participation.", 0),
    ("Hi James, following up on our conversation from yesterday, I've shared the Google Drive folder with the quarterly financial spreadsheets.", 0),
    ("Your appointment with Dr. Henderson has been confirmed for Tuesday, September 15 at 2:30 PM. Please arrive 10 minutes early to complete paperwork.", 0),
    ("Flight Confirmation: Your upcoming flight AA1042 to Chicago O'Hare is confirmed. Online check-in opens 24 hours prior to departure.", 0),
    ("Hi team, please welcome Priya who is joining our UX research team today as Senior Product Designer. Feel free to say hi on Slack!", 0),
    ("Code review feedback: Left a few comments on the database migration script. Overall looks solid, just check the indexing on the user_id column.", 0),
    ("Library Notice: Your borrowed book 'Designing Data-Intensive Applications' is due in 3 days. You can renew your loan online.", 0),
    ("Hi Mark, thanks for sharing the conference slides. Could you also email me the link to the recording when it gets uploaded?", 0),
    ("Payroll notification: Your direct deposit pay stub for the pay period ending August 31st is now accessible in the employee HR portal.", 0),
    ("Team lunch this Friday! Let's vote on Thai food or Mexican in the team chat before Thursday noon so we can make reservations.", 0),
    ("Invoice receipt from AWS: Your monthly cloud computing invoice for account ending in 4109 has been charged to your default payment card.", 0),
    ("Hi Michael, I tested the staging environment and all the API endpoints are responding correctly. Ready to deploy to production.", 0),
    ("University Alumni Newsletter: Read inspiring stories from our graduating class of 2024, upcoming homecoming events, and research breakthroughs.", 0),
    ("Hi team, please find the updated customer feedback survey results attached in PDF format. Overall satisfaction improved by 14%.", 0),
    ("Reminder to complete your annual mandatory cyber security awareness training module before the end of the month.", 0),
    ("Hotel booking confirmation: Your reservation at Grand Central Hotel for Oct 5-7 is confirmed. Free cancellation until Oct 3.", 0),
    ("Hi Susan, could you send over the high-resolution logos for the event banner? The print shop needs vector SVG or EPS files.", 0),
    ("Sprint planning notes: We will prioritize the search filter refactoring and the database schema migration during Sprint 34.", 0)
]

def train_and_evaluate():
    texts = [item[0] for item in TRAINING_DATA]
    labels = [item[1] for item in TRAINING_DATA]

    print(f"Total training samples: {len(texts)} (Spam: {sum(labels)}, Ham: {len(labels) - sum(labels)})")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    classifier = SpamClassifier()
    classifier.train(X_train, y_train)

    # Evaluate on test set
    y_pred = [1 if classifier.predict(t)["prediction"] == "Spam" else 0 for t in X_test]
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Evaluation ---")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    # Now train on full dataset for maximum production coverage
    full_classifier = SpamClassifier()
    full_classifier.train(texts, labels)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")
    full_classifier.save(save_path)
    print(f"Trained model saved successfully to: {save_path}")

    # Sanity check test
    test_spam = "URGENT: Click here now to claim your free $1000 prize lottery voucher!"
    res_spam = full_classifier.predict(test_spam)
    exp_spam = full_classifier.explain(test_spam)
    print(f"\nSanity Check (Spam Test): {res_spam}")
    print(f"Top Spam Factors: {[f['word'] for f in exp_spam['top_spam_words']]}")

    test_ham = "Hi Alex, please find the meeting agenda attached for tomorrow's team sync."
    res_ham = full_classifier.predict(test_ham)
    exp_ham = full_classifier.explain(test_ham)
    print(f"\nSanity Check (Ham Test): {res_ham}")
    print(f"Top Ham Factors: {[f['word'] for f in exp_ham['top_ham_words']]}")

if __name__ == "__main__":
    train_and_evaluate()
