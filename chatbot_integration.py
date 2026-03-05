class TravelChatbot:

    def __init__(self, engine):

        self.engine = engine


    def chat(self, message):

        message = message.lower()

        if "recommend" in message:
            return "I recommend exploring cultural heritage destinations."

        if "cost" in message:
            return "Average travel cost is around $150 per day."

        if "city" in message:
            return "Popular cities include Paris, Rome, Beijing, and Cusco."

        return "I can help you plan your travel itinerary."
