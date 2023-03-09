from sqlalchemy import (create_engine, Table,
                        MetaData, Column, String, Boolean, Integer, DateTime)
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# conn_string = "host='localhost' dbname='app_ind' user='user' password='postgres'"
engine = create_engine('postgresql+psycopg2://postgres:ffsfds&fdv12w@localhost:5432/datadb')
Base = declarative_base(bind=engine)

from sqlalchemy import (
    Column,
    Integer,
    String,
)


# class Indicators(Base):
#     __tablename__ = "reports_nlp_phrases"
#
#     report_name = Column(String(128), unique=False)
#     department = Column(String(128), unique=False)
#     indicators = Column(String(), unique=False)
#     name = Column(String(128), unique=False)
#     time_created = Column(DateTime(), default=datetime.utcnow)


class Users(Base):
    __tablename__ = "users_table"

    id = Column(Integer, primary_key=True)
    user_name = Column(String)
    email = Column(String)
    password = Column(String)
    company = Column(String)
    time_created = Column(DateTime(timezone=False), default=datetime.now().strftime("%m.%d.%Y %H:%M:%S"))


if __name__ == "__main__":
    Base.metadata.create_all(engine)