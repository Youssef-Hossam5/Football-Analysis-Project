from sklearn.cluster import KMeans


class TeamAssigner:
    """
    Assigns each player to a team based on their jersey color.

    Works in two stages:
        1. During setup (assign_team_color): samples all players from one frame,
           extracts their jersey colors, and clusters them into 2 teams.
        2. During tracking (get_player_team): for each new player seen, predicts
           which of the 2 team color clusters their jersey belongs to.
    """

    def __init__(self):
        # Stores the representative RGB color for each team {1: [R,G,B], 2: [R,G,B]}
        self.team_colors = {}

        # Cache of already-assigned players {player_id: team_id} to avoid re-predicting
        # the same player every frame
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        """
        Fits a 2-cluster KMeans model on an image's pixels.

        Reshapes the image from (height, width, 3) into a flat list of pixels
        (height*width, 3) so KMeans can treat each pixel as a data point.

        Args:
            image: A cropped image (numpy array of shape height x width x 3).

        Returns:
            kmeans: A fitted KMeans model with 2 clusters.
        """

        # Flatten image from (H, W, 3) → (H*W, 3)
        # KMeans expects a 2D array where each row is one [R, G, B] pixel
        image_2d = image.reshape(-1, 3)

        # k-means++ initialization spreads starting centroids apart,
        # giving more stable and accurate results than random initialization.
        # n_init=1 since k-means++ already gives a good starting point.
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Extracts the dominant jersey color of a single player.

        Only looks at the TOP HALF of the player's bounding box to focus on the
        jersey/shirt and avoid the shorts, socks, or grass at the bottom.

        Then clusters the top-half pixels into 2 groups and identifies which cluster
        is the jersey (not the background) by checking the 4 corners — corners are
        almost always background, not the player.

        Args:
            frame: Full video frame (BGR image).
            bbox:  Player's bounding box [x1, y1, x2, y2].

        Returns:
            player_color: RGB color of the player's jersey as [R, G, B].
        """

        # Crop the player out of the full frame using their bounding box
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Take only the top half — focuses on jersey, avoids shorts/grass
        top_half_image = image[0:int(image.shape[0] / 2), :]

        # Cluster the top half pixels into 2 color groups
        kmeans = self.get_clustering_model(top_half_image)

        # Get which cluster each pixel belongs to
        labels = kmeans.labels_

        # Reshape flat labels back into a 2D grid matching the image dimensions
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Check the 4 corners of the image — these are almost always background/grass
        corner_clusters = [
            clustered_image[0, 0],   # top-left
            clustered_image[0, -1],  # top-right
            clustered_image[-1, 0],  # bottom-left
            clustered_image[-1, -1]  # bottom-right
        ]

        # The most common corner cluster = background
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)

        # The other cluster (0 or 1) must be the player's jersey
        # Works because there are only 2 clusters, so 1 - 0 = 1 and 1 - 1 = 0
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        """
        Determines the two team colors by sampling all detected players in one frame.

        Extracts each player's jersey color, then clusters ALL those colors into 2 groups —
        one group per team. The resulting cluster centers become the team's reference colors.

        Args:
            frame:             Full video frame (BGR image).
            player_detections: Dict of {player_id: {'bbox': [x1, y1, x2, y2], ...}}
        """

        # Collect the jersey color of every detected player in this frame
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Cluster all player colors into 2 groups — one per team
        # n_init=10 here for more robust results since this is a one-time setup step
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
        kmeans.fit(player_colors)

        # Save the model for later use in get_player_team()
        self.kmeans = kmeans

        # Store each team's representative color (cluster center = average jersey color)
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Predicts which team a player belongs to based on their jersey color.

        Uses the KMeans model trained in assign_team_color() to classify the player's
        jersey color into one of the two team clusters.

        Results are cached in player_team_dict so each player is only classified once —
        subsequent calls for the same player_id return the cached result instantly.

        Args:
            frame:       Full video frame (BGR image).
            player_bbox: Player's bounding box [x1, y1, x2, y2].
            player_id:   Unique ID of the player.

        Returns:
            team_id: 1 or 2 representing which team the player belongs to.
        """

        # Return cached result if this player has been assigned before
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Extract this player's jersey color
        player_color = self.get_player_color(frame, player_bbox)

        # Predict which team cluster this color belongs to
        # reshape(1,-1) converts [R,G,B] → [[R,G,B]] as KMeans expects a 2D input
        # [0] extracts the single prediction from the returned array
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]

        # KMeans cluster IDs are 0 and 1 — shift to 1 and 2 to match our team numbering
        team_id += 1

        # Hardcoded override — player 91 is always assigned to team 1
        # (likely a goalkeeper or specific player that gets misclassified)
        if player_id == 91:
            team_id = 1

        # Cache the result so we don't re-predict this player in future frames
        self.player_team_dict[player_id] = team_id

        return team_id