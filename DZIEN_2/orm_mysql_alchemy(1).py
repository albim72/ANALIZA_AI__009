import sqlalchemy
import mysql.connector

from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import sessionmaker


engine = sqlalchemy.create_engine('mysql+mysqlconnector://root:abc123@localhost:3306/mojabazax',echo=True)

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = 'appusers'
    id = sqlalchemy.Column(sqlalchemy.Integer,primary_key=True)
    name = sqlalchemy.Column(sqlalchemy.String(length=50))
    fullname = sqlalchemy.Column(sqlalchemy.String(length=50))
    nickname = sqlalchemy.Column(sqlalchemy.String(length=50))

    def __repr__(self):
        return f'<User -> name: {self.name}, fullname: {self.fullname}, nickname: {self.nickname}>'

Base.metadata.create_all(engine)

Session = sessionmaker()
Session.configure(bind=engine)
session = Session()

us1 = User(name="Ola",fullname="Ola Nowak", nickname="olus")
session.add(us1)

us2 = User(name="Olaf",fullname="Olaf Kos", nickname="olo")
session.add(us2)

us3 = User(name="Tomasz",fullname="Tomasz Czarnik", nickname="czarny")
session.add(us3)

us4 = User(name="Maria",fullname="Maria Nowak", nickname="czarny")
session.add(us4)

session.commit()

print("_"*50)

for s in session.query(User).all():
    print(s)

print("_"*50)
for s in session.query(User).filter(User.nickname == 'czarny'):
    print(s.fullname)
