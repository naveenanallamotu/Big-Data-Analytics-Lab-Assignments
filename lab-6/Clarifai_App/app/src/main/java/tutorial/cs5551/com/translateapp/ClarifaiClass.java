package tutorial.cs5551.com.translateapp;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

import clarifai2.api.ClarifaiBuilder;
import clarifai2.api.ClarifaiClient;
import clarifai2.api.ClarifaiResponse;
import clarifai2.dto.input.ClarifaiInput;
import clarifai2.dto.input.image.ClarifaiImage;
import clarifai2.dto.model.output.ClarifaiOutput;
import clarifai2.dto.prediction.Concept;


/**
 * Created by gumma on 3/1/2017.
 */

public class ClarifaiClass {
    private final static String CLIENT_ID = "KozmDi2S-T_Pwy0Co0ebD2QVs9fNhdP-XE7quuoU";
    private final static String CLIENT_SECRET_KEY = "RUg0Fe5Yawnbh4z8I8G3DFf-rmTePxXS8bn4q54Y";

    public void predictImage(View v) {
        List<String> resultList = new ArrayList<String>();

                final ClarifaiClient client = new ClarifaiBuilder(CLIENT_ID, CLIENT_SECRET_KEY).buildSync();
                final List<ClarifaiOutput<Concept>> predictionResults =
                        client.getDefaultModels().generalModel() // You can also do client.getModelByID("id") to get custom models
                                .predict()
                                .withInputs(
                                        ClarifaiInput.forImage(ClarifaiImage.of("http://www.dream-wallpaper.com/free-wallpaper/nature-wallpaper/dream-homes-1-wallpaper/1280x800/free-wallpaper-9.jpg"))
                                )
                                .executeSync()
                                .get();
        if (predictionResults != null && predictionResults.size() > 0) {

            // Prediction List Iteration
            for (int i = 0; i < predictionResults.size(); i++) {

                ClarifaiOutput<Concept> clarifaiOutput = predictionResults.get(i);

                List<Concept> concepts = clarifaiOutput.data();

                if(concepts != null && concepts.size() > 0) {
                    for (int j = 0; j < concepts.size(); j++) {

                        resultList.add(concepts.get(j).name());
                    }
                }
            }
        }

    }

}





