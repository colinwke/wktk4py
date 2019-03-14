# date: 2019/3/14 10:44
# author: wang ke
# concat: ke.wang@ctrip.com
# ================================

"""python send email.
---
refs:
http://www.runoob.com/python3/python3-smtp.html
https://stackoverflow.com/a/6270987/6494418
https://blog.csdn.net/weixin_40475396/article/details/78693408
https://blog.csdn.net/gpf951101/article/details/78909233
https://blog.csdn.net/mouday/article/details/79896727
"""

import smtplib
from email.mime.text import MIMEText
from email.header import Header


class Email:
    def __init__(self):
        self.mail_host = "smtp.163.com"
        self.mail_user = "wfcrgt@163.com"
        self.mail_pass = "wang1ke23ctrip45"

        self.smtpObj = None

    def _login(self):
        self.smtpObj = smtplib.SMTP_SSL(self.mail_host, port=465)
        self.smtpObj.login(self.mail_user, self.mail_pass)

    def send_email(self, subject="none", content="none", receivers=None):
        if self.smtpObj is None:
            self._login()

        if receivers is None:
            receivers = self.mail_user

        message = MIMEText(content, "plain", "utf-8")
        message["Subject"] = Header(subject, "utf-8")
        message["From"] = self.mail_user  # same to sender
        message["To"] = str(receivers)

        print("\n".join(["=" * 32, str(message), "=" * 32]))
        self.smtpObj.sendmail(self.mail_user, receivers, message.as_string())
        print("send email success!")


if __name__ == '__main__':
    email = Email()

    email.send_email("python email test", "good test!")
