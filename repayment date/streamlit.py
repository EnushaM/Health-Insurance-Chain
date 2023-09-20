import streamlit as st
import pickle

def main():
    html_temp = """
    <div style="background-color:lightpink;padding:16px">
    <h2 style= "color:black";text-align:center> Health Insurance
    </div>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    model = pickle.loads(saved_model)
    print('Loaded model from disk')
    ins_code=st.text_input("Enter the Insurance Code")
    cus_no=st.number_input("Enter the Customer Number")
    cus_name=st.text_input("Enter the Name of the Customer")
    clear_date=st.date_input("Enter the Insurance Clear Date")
    year=st.number_input("Enter the Year of claiming Insurance ")
    ins_id=st.number_input("Enter the Unique Identifier of an Insurance")
    doc_id=st.date_input("Enter the  date on which the Insurance document was created")
    nor_doc_id=st.date_input("Enter the Normalised date of the Insurance document")
    due_date=st.date_input("Enter the date on which the customer is expected to clear an Insurance")
    curr=st.radio('Enter the Currency' ,['USD','CAD'])
    doc_type=st.radio('Enter the Document Type' ,['RV'])
    post_id=st.radio('Enter the posting ID',['1','2','3'])
    ins_date=st.date_input("Enter the date on which the Insurance was created")
    pay_typr=st.text_input("Enter the Business terms and agreements between customers and accounts on Insurance and days of payment")
    cus_ins_id=st.number_input("Enter the Unique number assigned when a person creates an Insurance.")
    ins_open=st.radio('Mention whether the Insurance is open or closed',['1','0'])
    if st.button('Predict'):
        pred=model.predict([[ins_code,cus_no,cus_name,clear_date,year,ins_id,doc_id,nor_doc_id,due_date,curr,doc_type,post_id,ins_date,pay_typr,cus_ins_id,ins_open]])
        pred=np.around(pred)
        delay=pd.Timedelta(days=pred)
        new_clear_date=due_date+delay
    
        if delay<=7:aging_bucket="0-7 days"
        elif delay<=15:aging_bucket"0-15 days"
        elif delay<=30:aging_bucket="16-30 days"
        elif delay<=45:aging_bucket="31-45 days"
        elif delay<=60:aging_bucket="46-60 days"
        else:aging_bucket="Greater than 60 days"

        st.success('The Possible Repayment Date is {} and the Delay period is {}'.format(new_clear_date,aging_bucket))